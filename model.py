import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from attention import *
from utils import *

class TrajectoryGenerator(nn.Module):
		def __init__(self, model_type="spatial_temporal", obs_len=8, pred_len=12, feature_dim=2, embedding_dim=16, encoder_dim=32, decoder_dim=32, attention_dim=32, domain_type="learnable", domain_parameter=5, delta_bearing=30, delta_heading=30, pretrained_scene="resnet18", device="cuda:0", noise_dim=None, noise_type=None):
				super(TrajectoryGenerator, self).__init__()
				self.obs_len=obs_len
				self.pred_len=pred_len
				self.model_type=model_type
				self.embedding_dim=embedding_dim
				self.decoder_dim=decoder_dim
				self.encoder_dim=encoder_dim
				self.device=device
				enc_count=0
				dec_count=0
				if ('generative' in self.model_type): 
						self.noise_dim=noise_dim
						self.noise_type=noise_type
				if ('spatial' in self.model_type):self.spatial_attention=spatial_attention(delta_bearing, delta_heading, domain_parameter, domain_type, attention_dim)
				if ('temporal' in self.model_type): self.temporal_attention=temporal_attention(encoder_dim, decoder_dim, attention_dim, obs_len)
				if not (embedding_dim==feature_dim): self.embedding = nn.Linear(feature_dim, embedding_dim)
				self.embedding_encoder = nn.Linear(2*attention_dim, encoder_dim)
				self.embedding_decoder = nn.Linear(2*attention_dim, decoder_dim) 
				self.encoder = nn.LSTMCell(embedding_dim, encoder_dim) 
				self.decoder = nn.LSTMCell(embedding_dim, decoder_dim)
				if ('generative' in self.model_type): self.hidden_mlp=nn.Linear(2*decoder_dim, decoder_dim) 
				#self.activation = nn.ReLU()
				self.activation = nn.Tanh()
				self.out = nn.Sequential(nn.Linear(decoder_dim, 2), nn.Tanh())
				if not (attention_dim==encoder_dim): self.encoder_attention_embedding = nn.Linear(encoder_dim, attention_dim)
				if not (attention_dim==decoder_dim): self.decoder_attention_embedding = nn.Linear(decoder_dim, attention_dim)
		def init_states(self, total_peds, hidden_dim):
				h_t, c_t = Variable(torch.zeros(total_peds, hidden_dim).to(self.device), requires_grad=True), Variable(torch.zeros(total_peds, hidden_dim).to(self.device), requires_grad=True)
				return h_t, c_t
		def encode(self, x, dmat, bmat, hmat, mask, batch_mean, batch_var, scene_embedding=None):
				batch_size, num_pedestrians = x.size()[:2]
				total_peds = batch_size*num_pedestrians
				h_t, c_t = self.init_states(total_peds, self.encoder_dim)
				encoded_input=[]
				for i, x_i in enumerate(x.chunk(x.size(2), dim=2)):
						x_i = x_i.squeeze(2).view(total_peds, -1)
						if hasattr(self, 'embedding'): x_i = self.embedding(x_i)
						h_t, c_t = self.encoder(x_i, (h_t, c_t))
						if hasattr(self, 'spatial_attention'): 
							h_t = self.spatial_attention(h_t, dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i])
						h_t = h_t.view(total_peds, -1)
						if hasattr(self, 'activation'): h_t = self.activation(h_t) 
						encoded_input+=[h_t.view(batch_size, num_pedestrians, 1, -1)]
				encoded_input = torch.stack(encoded_input, dim=2).view(batch_size, num_pedestrians, self.obs_len, -1)
				return encoded_input, h_t
		def decode(self, x, dmat, bmat, hmat, input_mask, output_mask, h_t, encoded_input, batch_mean, batch_var, scene_embedding=None):
				batch_mean, batch_var = batch_mean.squeeze(1), batch_var.squeeze(1)
				batch_size, num_pedestrians = x.size()[:2]
				total_peds = batch_size*num_pedestrians
				_, c_t = self.init_states(total_peds, self.decoder_dim)
				prev_x = x[:,:,-1,:]
				prev_dmat = dmat[:,:,-1,:]
				prev_bmat = bmat[:,:,-1,:]
				prev_hmat = hmat[:,:,-1,:]
				prev_mask = input_mask[:,:,-1]
				prediction=[]
				for j in range(self.pred_len):
					prev_x = prev_x.view(total_peds, -1) # (batch_size x num_pedestrians) x 2
					if hasattr(self, 'embedding'): prev_x_embedding = self.embedding(prev_x) # (batch_size x num_pedestrians) x embedding
					if hasattr(self, 'spatial_attention'):
						embedding = self.spatial_attention(h_t, prev_dmat, prev_bmat, prev_hmat, prev_mask)
						embedding = embedding.view(total_peds, -1)
						if hasattr(self, 'activation'): embedding = self.activation(embedding) 
						h_t, c_t = self.decoder(prev_x_embedding, (embedding, c_t))
					else:
						h_t, c_t = self.decoder(prev_x_embedding, (h_t.view(batch_size*num_pedestrians, -1), c_t))
					if hasattr(self, 'temporal_attention'):
						h_t, scores = self.temporal_attention(h_t.view(batch_size, num_pedestrians, -1), encoded_input, input_mask)
						if hasattr(self, 'activation'): h_t = self.activation(h_t)
					x_out = self.out(h_t).view(batch_size, num_pedestrians, -1)
					prev_dmat, prev_bmat, prev_hmat = get_features(x_out, 1, prev_x.view(batch_size, num_pedestrians, -1), mean=batch_mean, var=batch_var)
					prediction+=[x_out.unsqueeze(2)]
					prev_x = x_out
				prediction = torch.stack(prediction, dim=2)
				return prediction
		def forward(self, x, dmat, bmat, hmat, input_mask, output_mask, scene, batch_mean=None, batch_var=None):
			batch_size, num_pedestrians = x.size()[:2]
			total_peds = batch_size*num_pedestrians
			scene_embedding=None
			encoded_input, final_encoder_h = self.encode(x, dmat, bmat, hmat, input_mask, batch_mean, batch_var, scene_embedding)
			final_encoder_h = final_encoder_h.view(batch_size, num_pedestrians, -1)
			if ('generative' in self.model_type):
				#z = get_noise(final_encoder_h[0,...].size(), self.noise_type, self.device)
				#final_encoder_h = torch.cat([final_encoder_h.unsqueeze(2), z.repeat(batch_size,1,1).unsqueeze(2)], dim=2)
				# --changed this --
				z = get_noise(final_encoder_h[0,...].size(), self.noise_type, self.device)
				final_encoder_h = torch.cat([final_encoder_h.unsqueeze(2), z.repeat(batch_size,1, 1).unsqueeze(2)], dim=2)
				final_encoder_h = final_encoder_h.view(batch_size, num_pedestrians, -1)
			if hasattr(self, 'hidden_mlp'): final_encoder_h = self.hidden_mlp(final_encoder_h)
			prediction = self.decode(x, dmat, bmat, hmat, input_mask, output_mask, final_encoder_h, encoded_input, batch_mean, batch_var, scene_embedding)
			prediction = prediction.view(batch_size, num_pedestrians, self.pred_len, -1)
			return prediction 

def get_noise(shape, noise_type, device):
	if ("gaussian" in noise_type): return torch.randn(shape).to(device)
	elif ("uniform" in noise_type): return torch.rand(*shape).sub_(0.5).mul_(2.0).to(device)
	raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TrajectoryDiscriminator(nn.Module):
	def __init__(self, model_type, obs_len, pred_len, feature_dim=2, embedding_dim=16, encoder_dim=32, decoder_dim=32, attention_dim=32, domain_type="learnable", domain_parameter=5, delta_bearing=30, delta_heading=30):
		super(TrajectoryDiscriminator, self).__init__()
		self.obs_len=obs_len
		self.pred_len=pred_len
		self.embedding_dim=embedding_dim
		self.encoder_dim=encoder_dim
		self.device=device
		self.encoder=nn.LSTMCell(embedding_dim, encoder_dim)
		if not (encoder_dim==attention_dim):
			self.enc2att = nn.Linear(encoder_dim, attention_dim)
			self.att2enc = nn.Linear(attention_dim, encoder_dim)
		self.linear_embedding=nn.Linear(2, embedding_dim)
		mlp_dim=128
		if 'spatial' in model_type: self.spatial_attn=spatial_attention(delta_bearing, delta_heading, domain_parameter, domain_type, attention_dim)
		#self.classifier = nn.Linear(self.encoder_dim*(self.obs_len+self.pred_len), 1, bias=False)
		self.classifier = nn.Linear(self.encoder_dim, 1, bias=False)
	def init_states(self, batch_size, num_pedestrians, dim):
		h_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		c_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		return h_t, c_t
	def forward(self, traj, dmat, bmat, hmat, mask, output_mask):
		mask = torch.cat([mask, output_mask], dim=2)
		encoded_input, final_encoder_h = self.encode(traj, dmat, bmat, hmat, mask)
		scores = self.classifier(final_encoder_h)
		#scores = self.classifier(encoded_input)
		scores = scores.view(traj.size(0), traj.size(1))
		#scores = F.sigmoid(scores)
		#scores = F.softmax(scores)
		scores = F.softmax(scores, dim=-1)
		return scores.view(-1)
	def encode(self, x, dmat, bmat, hmat, mask):
		batch_size, num_pedestrians = x.size()[:2]
		h_t, c_t = self.init_states(batch_size, num_pedestrians, self.encoder_dim)
		encoded_input=[]
		for i, x_i in enumerate(x.chunk(x.size(2), dim=2)):
			x_i = x_i.squeeze(2)
			x_i = self.linear_embedding(x_i)
			h_t, c_t = self.encoder(x_i.view(batch_size*num_pedestrians, -1), (h_t, c_t))
			if hasattr(self, 'spatial_attn'): 
				h_t = self.spatial_attn(h_t.view(batch_size, num_pedestrians, -1),dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i])
				h_t = F.tanh(h_t).view(batch_size*num_pedestrians, -1)
			h_t = h_t.view(batch_size*num_pedestrians, -1)
			encoded_input+=[h_t.view(batch_size, num_pedestrians, 1, -1)]
		encoded_input = torch.stack(encoded_input, dim=2).view(batch_size, num_pedestrians, -1)
		return encoded_input, h_t




