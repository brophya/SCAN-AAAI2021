from __future__ import print_function

import sys
sys.dont_write_bytecode=True

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from model_utils import *


class TrajectoryGenerator(nn.Module):
	def __init__(self, model_type, obs_len, pred_len, ip_dim, op_dim, embedding_dim, encoder_dim, decoder_dim, attention_dim, device, domain_type="learnable", param_domain=5, delta_bearing=30, delta_heading=30, domain_init_type=None, noise_dim=None, noise_type=None, maxPeds=None):
		super(TrajectoryGenerator, self).__init__()
		self.obs_len=obs_len
		self.model_type=model_type
		self.pred_len=pred_len
		self.embedding_dim=embedding_dim
		self.encoder_dim=encoder_dim
		self.decoder_dim=decoder_dim
		self.device=device
		if ('generative' in self.model_type):
			self.noise_dim=noise_dim
			self.noise_type=noise_type
		if ('spatial' in self.model_type):
			if not (encoder_dim==attention_dim):
				self.enc2att = nn.Linear(encoder_dim, attention_dim)
				self.att2enc = nn.Linear(attention_dim, encoder_dim)
			if not (decoder_dim==attention_dim):
				self.dec2att=nn.Linear(decoder_dim, attention_dim)
				self.att2dec = nn.Linear(attention_dim, decoder_dim)
			if ('gat' in self.model_type): domain_type = 'gat'
			self.spatial_attn = spatial_attention(attention_dim,domain_type,param_domain,delta_bearing,delta_heading,domain_init_type,maxPeds)
		if ('temporal' in self.model_type):
			self.temporal_attn = temporal_attention(encoder_dim, decoder_dim, obs_len) 
		self.encoder=nn.LSTMCell(embedding_dim, encoder_dim)
		self.decoder=nn.LSTMCell(embedding_dim, decoder_dim)
		if ('generative' in self.model_type):
			self.mlp = nn.Linear(2*decoder_dim, decoder_dim)
		self.linear_out = nn.Sequential(nn.Linear(decoder_dim, 2), nn.Tanh())
		if not (embedding_dim==2):
			self.linear_embedding=nn.Linear(2, embedding_dim)
	def init_states(self, batch_size, num_pedestrians, dim):
		h_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		c_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		return h_t, c_t
	def encode(self, x, dmat, bmat, hmat, mask):
		batch_size, num_pedestrians = x.size()[:2]
		h_t, c_t = self.init_states(batch_size, num_pedestrians, self.encoder_dim)
		encoded_input=[]
		for i, x_i in enumerate(x.chunk(x.size(2), dim=2)):
			x_i = x_i.squeeze(2)
			if hasattr(self, 'linear_embedding'): x_i = self.linear_embedding(x_i)
			if ('spatial' in self.model_type):
				if hasattr(self, 'enc2att'):
					h_t_emb = self.enc2att(h_t.view(batch_size, num_pedestrians, -1))
					w_h = self.spatial_attn(h_t_emb, dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i])
				else:
					w_h = self.spatial_attn(h_t.view(batch_size, num_pedestrians, -1),dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i])
				if hasattr(self, 'att2enc'):
					w_h = self.att2enc(w_h).view(batch_size*num_pedestrians, -1)
				else:
					w_h = w_h.view(batch_size*num_pedestrians, -1)
				h_t, c_t = self.encoder(x_i.view(batch_size*num_pedestrians, -1), (w_h, c_t))
				encoded_input+=[w_h.view(batch_size, num_pedestrians, 1, -1)]
			else:
				h_t, c_t = self.encoder(x_i.view(batch_size*num_pedestrians, -1), (h_t, c_t))
				encoded_input+=[h_t.view(batch_size, num_pedestrians, 1, -1)]
		encoded_input = torch.stack(encoded_input, dim=2).view(batch_size, num_pedestrians, self.obs_len, -1)
		return encoded_input, h_t
	def decode(self, x, dmat, bmat, hmat, ip_mask, op_mask, h_t, encoded_input):
		batch_size, num_pedestrians = x.size()[:2]
		_, c_t = self.init_states(batch_size, num_pedestrians, self.decoder_dim)
		x_prev = x[:,:,-1,:]
		dmat_ = dmat[:,:,-1,:]
		bmat_ = bmat[:,:,-1,:]
		hmat_ = hmat[:,:,-1,:]
		mask_ = ip_mask[:,:,-1]
		prediction = []
		for j in range(self.pred_len):
			if hasattr(self, 'linear_embedding'):
				x_emb = self.linear_embedding(x_prev)
			else:
				x_emb = x_prev
			if ('spatial' in self.model_type):
				if hasattr(self, 'dec2att'):
					h_t_emb = self.dec2att(h_t)
					w_h = self.spatial_attn(h_t_emb, dmat_, bmat_, hmat_, mask_)
				else:
					w_h = self.spatial_attn(h_t, dmat_, bmat_, hmat_, mask_)
				if hasattr(self, 'att2dec'):
					w_h = self.att2dec(w_h)
				if ('temporal' in self.model_type):
					w_h, alignment_vector = self.temporal_attn(w_h.view(batch_size, num_pedestrians, -1), encoded_input,  ip_mask)
				h_t, c_t = self.decoder(x_emb.view(batch_size*num_pedestrians, -1), (w_h.view(batch_size*num_pedestrians, -1), c_t))
			else:
				if ('temporal' in self.model_type):
					h_t, alignment_vector = self.temporal_attn(h_t.view(batch_size, num_pedestrians, -1), encoded_input,  ip_mask)
				h_t, c_t = self.decoder(x_emb.view(batch_size*num_pedestrians, -1), (h_t.view(batch_size*num_pedestrians, -1), c_t))
			h_t = h_t.view(batch_size, num_pedestrians, -1)
			x_out = self.linear_out(h_t.view(batch_size, num_pedestrians, -1)) 
			dmat_, bmat_, mat_ = get_features(x_out, 1, x_prev)
			prediction+=[x_out.unsqueeze(2)]
			x_prev=x_out
		prediction=torch.stack(prediction, dim=2)
		return prediction
	def forward(self, x, dmat, bmat, hmat, input_mask, output_mask):
		batch_size, num_pedestrians = x.size()[:2]
		encoded_input, final_encoder_h = self.encode(x, dmat, bmat, hmat, input_mask)
		final_encoder_h = final_encoder_h.view(batch_size, num_pedestrians, -1)
		if ('generative' in self.model_type):
			z = get_noise(final_encoder_h[0,...].size(), self.noise_type, self.device)
			final_encoder_h = torch.cat([final_encoder_h.unsqueeze(2), z.repeat(batch_size, 1, 1).unsqueeze(2)], dim=2)
			final_encoder_h = final_encoder_h.view(batch_size, num_pedestrians, -1)
			final_encoder_h = self.mlp(final_encoder_h) 
		prediction = self.decode(x, dmat, bmat, hmat, input_mask, output_mask, final_encoder_h, encoded_input)
		prediction = prediction.view(batch_size, num_pedestrians, self.pred_len, -1)
		return prediction


def get_noise(shape, noise_type, device):
	if noise_type=="gaussian":
		return torch.randn(shape).to(device)
	elif noise_type=="uniform":
		return torch.rand(*shape).sub_(0.5).mul_(2.0).to(device)
	raise ValueError('Unrecognized noise type "%s"' % noise_type)

class TrajectoryDiscriminator(nn.Module):
	def __init__(self, sequence_length=8, prediction_length=12, embedding_dim=16, encoder_dim=16, delta_bearing=30, delta_heading=30, attention_dim=16, mlp_dim=128, dropout=0.0, domain_type="learnable", param_domain=5, domain_init_type="constant", device=None):
		super(TrajectoryDiscriminator, self).__init__()
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.embedding_dim=embedding_dim
		self.encoder_dim=encoder_dim
		self.device=device
		self.encoder=nn.LSTMCell(embedding_dim, encoder_dim)
		if not (encoder_dim==attention_dim):
			self.enc2att = nn.Linear(encoder_dim, attention_dim)
			self.att2enc = nn.Linear(attention_dim, encoder_dim)
		self.spatial_attn = spatial_attention(attention_dim,domain_type,param_domain,delta_bearing,delta_heading,domain_init_type)
		self.linear_embedding=nn.Linear(2, embedding_dim)
		self.classifier = nn.Sequential(nn.Linear((self.encoder_dim, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, 1), nn.LeakyReLU())
	def init_states(self, batch_size, num_pedestrians, dim):
		h_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		c_t = Variable(torch.zeros(batch_size*num_pedestrians, dim).to(self.device), requires_grad=True)
		return h_t, c_t
	def forward(self, traj, dmat, bmat, hmat, mask, output_mask):
		mask = torch.cat([mask, output_mask], dim=2)
		encoded_input, final_encoder_h = self.encode(traj, dmat, bmat, hmat, mask)
		scores = self.classifier(final_encoder_h) 
		scores = scores.view(traj.size(0), traj.size(1))
		scores = F.softmax(scores, dim=1)
		return scores.view(-1)
	def encode(self, x, dmat, bmat, hmat, mask):
		batch_size, num_pedestrians = x.size()[:2]
		h_t, c_t = self.init_states(batch_size, num_pedestrians, self.encoder_dim)
		encoded_input=[]
		for i, x_i in enumerate(x.chunk(x.size(2), dim=2)):
			x_i = x_i.squeeze(2)
			x_i = self.linear_embedding(x_i)
			w_h = self.spatial_attn(h_t.view(batch_size, num_pedestrians, -1),dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i])
			w_h = w_h.view(batch_size*num_pedestrians, -1)
			h_t, c_t = self.encoder(x_i.view(batch_size*num_pedestrians, -1), (w_h, c_t))
			encoded_input+=[h_t.view(batch_size, num_pedestrians, -1)]
		encoded_input = torch.stack(encoded_input, dim=2)
		return encoded_input, h_t




