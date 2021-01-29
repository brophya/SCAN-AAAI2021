import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

from utils import *

def masked_softmax(vec, dim=1, epsilon=1e-14):
    mask = ~(vec==0)
    mask = mask.bool()
    exps = torch.exp(vec)
    masked_exps = exps * mask.float() + epsilon 
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

class spatial_attention(nn.Module):
    def __init__(self, delta_bearing, delta_heading, domain_parameter, domain_type, attention_dim):
        super(spatial_attention, self).__init__()
        self.delta_bearing=delta_bearing
        self.delta_heading=delta_heading
        self.domain=Variable(torch.FloatTensor(int(360/delta_heading), int(360/delta_bearing)), requires_grad=True)
        self.domain=nn.Parameter(self.domain)
        nn.init.constant_(self.domain.data, domain_parameter)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(2*attention_dim, attention_dim)
    def forward(self, hidden_state, distance_matrix, bearing_matrix, heading_matrix, sequence_mask):
        weights = self.compute_weights(distance_matrix, bearing_matrix, heading_matrix, sequence_mask)
        weighted_hidden = torch.bmm(weights, hidden_state.view(weights.size(0), weights.size(1), -1))
#        weighted_hidden = self.tanh(weighted_hidden)
        weighted_hidden = torch.cat((weighted_hidden, hidden_state.view_as(weighted_hidden)), dim=-1)
        weighted_hidden = self.linear(weighted_hidden)
        return weighted_hidden
    def compute_weights(self, distance_matrix, bearing_matrix, heading_matrix, sequence_mask):
        batch_size, num_pedestrians, num_pedestrians = distance_matrix.size()
        idx1, idx2 = torch.floor((heading_matrix+(self.delta_heading/2))/self.delta_heading), torch.floor((bearing_matrix+(self.delta_bearing/2))/self.delta_bearing)
	#idx1, idx2 = torch.floor(heading_matrix/self.delta_heading), torch.floor(bearing_matrix/self.delta_bearing)
        idx1, idx2 = idx1.clamp(0, int(360/self.delta_heading)-1), idx2.clamp(0, int(360/self.delta_bearing)-1)
        weights=self.relu(self.domain[idx1.long(), idx2.long()]-distance_matrix)
        weights_mask = sequence_mask.unsqueeze(-1).expand(distance_matrix.size())
        weights_mask = weights_mask.mul(weights_mask.transpose(1,2))
        ped_ix = range(weights.size(1))
        ped_mask = torch.ones_like(weights)
        ped_mask[:, ped_ix, ped_ix] = 0
        weights.data.masked_fill_(mask=~(ped_mask.bool()*weights_mask.bool()), value=float(0))
        weights = masked_softmax(weights, dim=2)
        weights.data.masked_fill_(mask=~(ped_mask.bool()*weights_mask.bool()), value=float(0))
        return weights 

class temporal_attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, obs_len):
        super(temporal_attention, self).__init__()
        self.obs_len=obs_len
        self.attention_dim=attention_dim
        self.linear=nn.Linear(2*decoder_dim, decoder_dim)
        self.softmax=nn.Softmax(dim=1)
        if not (encoder_dim==attention_dim): self.encoder_embedding=nn.Linear(encoder_dim, attention_dim)
        if not (decoder_dim==attention_dim): self.decoder_embedding=nn.Linear(decoder_dim, attention_dim)
    def compute_score(self, hidden_encoder, hidden_decoder, sequence_mask):
        # check this
        score = torch.bmm(hidden_encoder, hidden_decoder.unsqueeze(-1)).squeeze(-1)
        score.data.masked_fill_(mask=~sequence_mask.view(-1, self.obs_len), value=float(-1e24))
        score = self.softmax(score)
        return score 
    def forward(self, hidden_decoder, hidden_encoder, sequence_mask):
        if hasattr(self, 'encoder_embedding'): encoder_embedding=self.encoder_embedding(hidden_encoder)
        else: encoder_embedding=hidden_encoder 
        if hasattr(self, 'decoder_embedding'): decoder_embedding=self.decoder_embedding(hidden_decoder)
        else: decoder_embedding=hidden_decoder
        encoder_embedding = encoder_embedding.view(-1, self.obs_len, self.attention_dim)
        decoder_embedding = decoder_embedding.view(-1, self.attention_dim)
        score=self.compute_score(encoder_embedding, decoder_embedding, sequence_mask)
        context_vector=torch.bmm(score.unsqueeze(1), encoder_embedding).squeeze(1)
 #       context_vector=F.tanh(context_vector)
        out = torch.cat((context_vector, hidden_decoder.view_as(context_vector)), dim=-1)
        out = self.linear(out)
        return out, score 
        
class scene_attention(nn.Module):
    def __init__(self, model, attention_dim):
        super(scene_attention, self).__init__()
        self.net = getattr(models, model)(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:4])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        for param in self.net.parameters(): param.requires_grad=False
        self.fc=nn.Linear(64, attention_dim)
        self.tanh = nn.Tanh()
    def forward(self, ip_img):
        features = self.net(ip_img)
        features = self.pool(features)
        features = self.fc(features.view(features.size(0), -1))
        return features 
        

