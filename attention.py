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
	masked_exps = exps * mask.float() #+ epsilon
	masked_sums = masked_exps.sum(dim, keepdim=True) #+ epsilon
	mask_ = masked_sums.expand_as(masked_exps)
	zeros_ = torch.zeros_like(mask_)
	out = torch.where(~(mask_==0), masked_exps/(masked_sums+epsilon), zeros_) 
	return out

class spatial_attention(nn.Module):
	def __init__(self, delta_bearing, delta_heading, domain_parameter, attention_dim):
		super(spatial_attention, self).__init__()
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.domain=Variable(torch.FloatTensor(int(360/delta_heading), int(360/delta_bearing)), requires_grad=True)
		self.domain=nn.Parameter(self.domain)
		nn.init.constant_(self.domain, domain_parameter)
		self.relu = nn.ReLU()
		self.attention_dim=attention_dim
	def forward(self, hidden_state, distance_matrix, bearing_matrix, heading_matrix, sequence_mask, domain=None):
		batch_size, num_pedestrians, num_pedestrians = distance_matrix.size()
		total_peds=batch_size*num_pedestrians
		weights = self.compute_weights(distance_matrix, bearing_matrix, heading_matrix, sequence_mask, domain)
		weighted_hidden = torch.bmm(weights, hidden_state)
		weighted_hidden = weighted_hidden.view(total_peds, self.attention_dim)
		weighted_hidden = torch.cat((weighted_hidden, hidden_state.view_as(weighted_hidden)), dim=1)
		return weighted_hidden
	def compute_weights(self, distance_matrix, bearing_matrix, heading_matrix, sequence_mask, domain=None):
		batch_size, num_pedestrians, num_pedestrians = distance_matrix.size()
		total_peds=batch_size*num_pedestrians
		idx1, idx2 = self.convert_to_bins(bearing_matrix, heading_matrix) 
		if not domain is None:
			weights=self.relu(domain[idx1, idx2]-distance_matrix)
		else:
			weights=self.relu(self.domain[idx1, idx2]-distance_matrix)
		weights_mask = sequence_mask.unsqueeze(-1).expand(distance_matrix.size())
		weights_mask = weights_mask.mul(weights_mask.transpose(1,2))
		self_ped = torch.ones_like(weights)
		self_ped[:, range(num_pedestrians), range(num_pedestrians)] = 0
		weights.data.masked_fill_(mask=~weights_mask.bool(), value=float(0))
		weights.data.masked_fill_(mask=~self_ped.bool(), value=float(0))
		weights=masked_softmax(weights, dim=2)
		return weights
	def convert_to_bins(self, bearing_matrix, heading_matrix):
		shifted_heading=(heading_matrix+(self.delta_heading/2))
		shifted_bearing=(bearing_matrix+(self.delta_bearing/2))
		shifted_heading=torch.where(shifted_heading>=360, shifted_heading-360, shifted_heading)
		shifted_bearing=torch.where(shifted_bearing>=360, shifted_bearing-360, shifted_bearing)
		idx1, idx2 = torch.floor(shifted_heading/self.delta_heading), torch.floor(shifted_bearing/self.delta_bearing)
		idx1, idx2 = torch.clamp(idx1, 0, int(360/self.delta_heading)-1), torch.clamp(idx2, 0, int(360/self.delta_bearing)-1)
		idx1, idx2 = idx1.long(), idx2.long()
		return idx1, idx2
class temporal_attention(nn.Module):
	def __init__(self, encoder_dim, decoder_dim, attention_dim, obs_len):
		super(temporal_attention, self).__init__()
		self.obs_len=obs_len
		self.attention_dim=attention_dim
		self.linear=nn.Sequential(nn.Linear(2*attention_dim, decoder_dim),nn.Tanh())
		self.softmax=nn.Softmax(dim=1)
		self.encoder_embedding=nn.Linear(encoder_dim, attention_dim)
		self.decoder_embedding=nn.Linear(decoder_dim, attention_dim)
	def compute_score(self, hidden_encoder, hidden_decoder, sequence_mask):
		score = torch.bmm(hidden_encoder, hidden_decoder.unsqueeze(-1)).squeeze(-1)
		score.data.masked_fill_(mask=~sequence_mask.view(-1, self.obs_len), value=float(-1e24))
		score = self.softmax(score)
		return score
	def forward(self, hidden_decoder, hidden_encoder, sequence_mask):
		hidden_encoder=hidden_encoder.squeeze(2)
		if hasattr(self, 'encoder_embedding'): encoder_embedding=self.encoder_embedding(hidden_encoder)
		else: encoder_embedding=hidden_encoder 
		if hasattr(self, 'decoder_embedding'): decoder_embedding=self.decoder_embedding(hidden_decoder)
		else: decoder_embedding=hidden_decoder
		score=self.compute_score(encoder_embedding, decoder_embedding, sequence_mask)
		context_vector=torch.bmm(score.unsqueeze(1), encoder_embedding).squeeze(1)
		out = torch.cat((context_vector, decoder_embedding.view_as(context_vector)), dim=-1)
		out = self.linear(out)
		return out, score 
        
class scene_attention_(nn.Module):
	def __init__(self, model, attention_dim):
		super(scene_attention_, self).__init__()
		self.net = getattr(models, model)(pretrained=True)
		self.net = nn.Sequential(*list(self.net.children())[:4])
		self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
		for param in self.net.parameters(): param.requires_grad=False
		self.tanh=nn.Tanh()
		self.fc=nn.Sequential(nn.Linear(64, attention_dim), nn.Tanh())
	def forward(self, ip_img):
		features = self.net(ip_img)
		features = self.tanh(features)
		features = self.pool(features)
		features = self.fc(features.view(features.size(0), -1))
		return features 


