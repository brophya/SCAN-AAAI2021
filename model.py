import torch
import torch.nn as nn

from torch.autograd import Variable
from attention import *
from utils import *

class lstm(nn.Module):
	def __init__(self, ip_dim, hidden_dim):
		super(lstm, self).__init__()
		self.lstm=nn.LSTM(ip_dim, hidden_dim, batch_first=True)
		self.ip_dim=ip_dim
		self.hidden_dim=hidden_dim
	def forward(self, x, h_t, c_t):
		"""
		x -> batch_size x num_pedestrians x embedding_dim 

		since nn.LSTM does not take 3D input, 
		x is reshaped as [batch_size x num_pedestrians] x embedding_dim 
		before feeding to the nn.LSTM model
		"""
		batch_size, num_pedestrians = list(x.size())[:2]
		total_peds = batch_size*num_pedestrians
		x = x.view(total_peds,1, self.ip_dim)
		h_t = h_t.view(1, total_peds, self.hidden_dim)
		_ , (h_t, c_t) = self.lstm(x, (h_t,c_t))
		h_t = h_t.squeeze(0).view(batch_size, num_pedestrians, self.hidden_dim)
		return h_t, c_t


class TrajectoryGenerator(nn.Module):
	def __init__(self, model_type="spatial_temporal", obs_len=8, pred_len=12, feature_dim=2, embedding_dim=16, encoder_dim=32, decoder_dim=32, attention_dim=32, domain_parameter=5, delta_bearing=30, delta_heading=30, pretrained_scene="resnet18", device="cuda:0", noise_dim=None, noise_type=None, noise_mix_type='sample', dropout=0.2):
		
		super(TrajectoryGenerator, self).__init__()
		
		self.obs_len=obs_len
		self.pred_len=pred_len
		self.model_type=model_type
		self.embedding_dim=embedding_dim
		self.decoder_dim=decoder_dim
		self.encoder_dim=encoder_dim
		self.attention_dim=attention_dim

		self.device=device
	
		self.encoder_embedding=nn.Linear(feature_dim, embedding_dim)
		self.decoder_embedding=nn.Linear(feature_dim, embedding_dim)

		self.encoder=lstm(embedding_dim, encoder_dim)
		self.decoder=lstm(embedding_dim, decoder_dim)
		
		if ('spatial' in self.model_type):
			self.spatial_attention=spatial_attention(delta_bearing, delta_heading, domain_parameter, attention_dim)
			n=2
			self.decoder_spatial_embedding=nn.Sequential(nn.Linear(n*attention_dim, attention_dim), nn.Tanh())
			self.encoder_spatial_embedding=nn.Sequential(nn.Linear(n*attention_dim, attention_dim), nn.Tanh())

		if ('temporal' in self.model_type):
			self.temporal_attention=temporal_attention(encoder_dim, decoder_dim, attention_dim, obs_len)

		if ('generative' in self.model_type): 
			self.noise_type=noise_type
			self.noise_mix_type=noise_mix_type
			
			if noise_dim is None:
				self.noise_dim=encoder_dim//2
				self.noise_mlp=nn.Linear(encoder_dim, encoder_dim//2)
			
			else:
				self.noise_dim=noise_dim
				if not (noise_dim==encoder_dim):
					self.noise_mlp=nn.Linear(encoder_dim, encoder_dim-noise_dim)

		self.out = nn.Sequential(nn.Linear(decoder_dim, 2), nn.ReLU())

		self.att2enc=nn.Linear(attention_dim, encoder_dim)
		self.enc2att=nn.Linear(encoder_dim, attention_dim)
		
		self.att2dec=nn.Linear(attention_dim, decoder_dim)
		self.dec2att=nn.Linear(decoder_dim, attention_dim)
			
		if not (encoder_dim==decoder_dim):
			self.enc2dec = nn.Linear(encoder_dim, decoder_dim)

		
	def init_states(self, total_peds, hidden_dim):
		h_t = Variable(torch.zeros(total_peds, hidden_dim).to(self.device), requires_grad=True)
		c_t = Variable(torch.zeros(1, total_peds, hidden_dim).to(self.device), requires_grad=True)
		return h_t, c_t
	def encode(self, x, dmat, bmat, hmat, mask, mean, var, scene, domain=None):
		"""
		x -> batch_size x num_pedestrians x observation_length x 2
		dmat -> batch_size x num_pedestrians x observation_length x num_pedestrians
		bmat -> batch_size x num_pedestrians x observation_length x num_pedestrians
		hmat -> batch_size x num_pedestrians x observation_length x num_pedestrians
		mask -> batch_size x num_pedestrians x observation_length 
		mean, var: normalizing parameters 
		"""
		batch_size, num_pedestrians = x.size()[:2]
		h_t, c_t = self.init_states(batch_size * num_pedestrians, self.encoder_dim) 
		embedded_x = self.encoder_embedding(x)
		encoded_input=[]
		for i, x_i in enumerate(embedded_x.chunk(x.size(2), dim=2)):
			h_t, c_t = self.encoder(x_i, h_t, c_t)
			if hasattr(self, 'spatial_attention'):
				if hasattr(self, 'enc2att'): h_t = self.enc2att(h_t) 
				if hasattr(self, 'act'): h_t=self.act(h_t)
				h_t = self.spatial_attention(h_t, dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i], domain=domain)
				
				h_t = self.encoder_spatial_embedding(h_t) 
				if hasattr(self, 'att2enc'):h_t = self.att2enc(h_t)
				if hasattr(self, 'act'): h_t=self.act(h_t)
			encoded_input+=[h_t]
		return h_t, encoded_input
	def decode(self, x, distance_matrix, bearing_matrix, heading_matrix, input_mask, output_mask, h_t, encoded_input, mean, var, scene, domain=None):
		"""
		x-> last observed trajectory position for all num_pedestrians 
		distance_matrix-> last observed distances for all num_pedestrians 
		bearing_matrix-> last observed relative bearing for all num_pedestrians
		heading_matrix-> last observed relative heading for all num_pedestrians
		input_mask -> mask for last observed positions 
		output_mask-> mask for predicted trajectories
		h_t -> encoder's last hidden state for all num_pedestrians 
		encoded_input -> spatially weighted hidden states from the observed trajectory
		mean, var -> normalizing parameters
		"""
		if self.training: eps=1e-14
		else: eps=0
		mean, var = mean.squeeze(1), var.squeeze(1)
		batch_size, num_pedestrians = x.size()[:2]
		_, c_t = self.init_states(batch_size * num_pedestrians, self.decoder_dim) 
		encoded_input=torch.stack(encoded_input, dim=2)
		prediction=[]
		x_ = revert_orig_tensor(x, mean, var, input_mask[:,:,-1], dim=1)
		for j in range(self.pred_len):
			embedded_x=self.decoder_embedding(x)
			h_t, c_t = self.decoder(embedded_x, h_t, c_t)
			if hasattr(self, 'spatial_attention'):
				if hasattr(self, 'dec2att'): h_t=self.dec2att(h_t) 
				if hasattr(self, 'act'): h_t=self.act(h_t)
				h_t=self.spatial_attention(h_t,distance_matrix,bearing_matrix,heading_matrix,output_mask[:,:,j],domain=domain)
				
				h_t=self.decoder_spatial_embedding(h_t) 
				if hasattr(self, 'att2dec'): h_t=self.att2dec(h_t) 
				if hasattr(self, 'act'): h_t=self.act(h_t)
			if hasattr(self, 'temporal_attention'):
				h_t, scores=self.temporal_attention(h_t, encoded_input, input_mask) 
			x_out=self.out(h_t) 
			x_out = x_out * output_mask[:,:,j].unsqueeze(-1).expand_as(x_out)
			x_out_ = revert_orig_tensor(x_out, mean, var, output_mask[:,:,j], dim=1)
			distance_matrix, bearing_matrix, heading_matrix=get_features(x_out_, 1, x_, mask=output_mask[:,:,j], eps=eps) 
			prediction+=[x_out]
			x_ = x_out_
			x = x_out
		prediction = torch.stack(prediction, dim=2)
		return prediction
	def attend_to_scene(self, last_embedding, h_t, scene):
		"""
		#### Not used in AAAI version ######
		computes scene embedding 
		"""
		batch_size, num_pedestrians = list(last_embedding.size())[:2]
		h_scene = self.scene_attention(scene, last_embedding)
		return h_scene
	def add_noise(self, h_t, mask):
		"""
		Adds noise to final output of encoder for all num_pedestrians 
		noise_mix_type ped adds the same noise to all pedestrians in a sample 
		noise_mix_type sample adds different noise to all pedestrians in a sample
		"""
		batch_size, num_pedestrians, h_dim = h_t.size()
		if hasattr(self, 'noise_mlp'): h_t=self.noise_mlp(h_t)
		if self.noise_mix_type=='ped':
			z = get_noise((batch_size, 1, self.noise_dim), self.noise_type, self.device)
			z = z.repeat(1, num_pedestrians, 1)
		elif self.noise_mix_type=='sample':
			z = get_noise((batch_size, num_pedestrians, self.noise_dim), self.noise_type, self.device)
		if self.noise_dim!=self.encoder_dim:
			h_t = torch.cat((h_t, z), dim=2)
		else:
			h_t=z 
		return h_t 
	def forward(self, x, pedestrians, dmat, bmat, hmat, input_mask, output_mask, scene=None, mean=None, var=None, domain=None):
		batch_size, num_pedestrians = x.size()[:2]
		final_h, encoded_input = self.encode(x, dmat, bmat, hmat, input_mask, mean, var, scene, domain=domain)
		if ('generative' in self.model_type):
			final_h = self.add_noise(final_h, input_mask[:,:,-1])
		if hasattr(self, 'enc2dec'): final_h = self.enc2dec(final_h)
		prediction=self.decode(x[:,:,-1,:], dmat[:,:,-1,:], bmat[:,:,-1,:], hmat[:,:,-1,:], input_mask, output_mask, final_h, encoded_input, mean, var, scene, domain=domain)
		return prediction

def get_noise(shape, noise_type, device):
	if ("gaussian" in noise_type): return torch.randn(shape).to(device)
	elif ("uniform" in noise_type): return torch.rand(*shape).mul_(2.0).sub_(1.0).to(device)
	raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TrajectoryDiscriminator(nn.Module):
	def __init__(self, model_type='lstm', seq_len=20, feature_dim=2, embedding_dim=16, hidden_size=32, mlp_dim=128, attention_dim=32, delta_bearing=30, delta_heading=30, domain_parameter=5):
		super(TrajectoryDiscriminator,self).__init__()
		self.hidden_size=hidden_size
		self.num_layers=1
		self.seq_len=seq_len
		self.embedding_dim=embedding_dim
		if model_type=='lstm':
			self.encoder=lstm(embedding_dim, hidden_size)
		elif model_type=='fc':
			self.fc = nn.Sequential(nn.Linear(self.embedding_dim*self.seq_len, mlp_dim), nn.LeakyReLU(negative_slope=0.2), nn.Linear(mlp_dim, mlp_dim), nn.LeakyReLU(negative_slope=0.2), nn.Linear(mlp_dim, hidden_size), nn.LeakyReLU(negative_slope=0.2))
		self.embedding=nn.Linear(feature_dim, embedding_dim)
		self.encoder_spatial_embedding=nn.Sequential(nn.Linear(2*attention_dim, attention_dim), nn.Tanh())


		self.spatial_attention=spatial_attention(delta_bearing, delta_heading, domain_parameter, attention_dim)
		
		self.classifier = nn.Sequential(nn.Linear(self.seq_len*hidden_size, mlp_dim), nn.LeakyReLU(negative_slope=0.2), nn.Linear(mlp_dim, mlp_dim), nn.LeakyReLU(negative_slope=0.2), nn.Linear(mlp_dim, 1))

		self.enc2att=nn.Linear(hidden_size, attention_dim)
		self.att2enc=nn.Linear(attention_dim, hidden_size)

		self.sigmoid=nn.Sigmoid()
	def init_hidden(self, total_peds):
		return (Variable(torch.zeros(self.num_layers, total_peds, self.hidden_size).to("cuda:0"), requires_grad=True),
			Variable(torch.zeros(self.num_layers, total_peds, self.hidden_size).to("cuda:0"), requires_grad=True)
			)
	def forward(self, x, dmat=None, bmat=None, hmat=None, mask=None, domain=None):
		x = self.embedding(x)
		batch_size, num_pedestrians = x.size()[:2]
		h_t, c_t = self.init_hidden(batch_size * num_pedestrians)
		encoded_input=[]
		if not dmat is None:
			for i in range(self.seq_len):
				x_i = x[:,:,i,:]
				h_t, c_t = self.encoder(x_i, h_t, c_t)
				if hasattr(self, 'spatial_attention'):
					if hasattr(self, 'enc2att'):
						h_t = self.enc2att(h_t)
					h_t=self.spatial_attention(h_t,dmat[:,:,i,:],bmat[:,:,i,:],hmat[:,:,i,:],mask[:,:,i], domain)
					h_t = self.encoder_spatial_embedding(h_t)
					if hasattr(self, 'att2enc'):
						h_t = self.att2enc(h_t)
				encoded_input+=[h_t]
		else:
			if hasattr(self, 'encoder'):
				for i in range(self.seq_len):
					h_t, c_t = self.encoder(x[:,:,i,:], h_t, c_t)
					encoded_input+=[h_t]
			elif hasattr(self, 'fc'):
				h_t = self.fc(x.view(batch_size, num_pedestrians, self.embedding_dim*self.seq_len))
		
		encoded_input=torch.cat(encoded_input, 2).view(batch_size, num_pedestrians, self.seq_len*self.hidden_size).to("cuda:0")
		scores = self.classifier(encoded_input)
		scores = self.sigmoid(scores)
		return scores



