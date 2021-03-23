import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from attention import *
from utils import *
from gat_utils import *

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
	
		self.encoder_embedding=nn.Sequential(nn.Linear(feature_dim, embedding_dim, bias=False), nn.ReLU())
		self.decoder_embedding=nn.Sequential(nn.Linear(feature_dim, embedding_dim, bias=False), nn.ReLU())

		self.encoder = nn.LSTM(embedding_dim, encoder_dim, batch_first=True) 
		self.decoder = nn.LSTM(embedding_dim, decoder_dim, batch_first=True)

		if ('spatial' in self.model_type):
			self.spatial_attention = GraphAttentionLayer(attention_dim, attention_dim)

		if ('temporal' in self.model_type):
			self.temporal_attention=temporal_attention(encoder_dim, decoder_dim, attention_dim, obs_len)

		if ('generative' in self.model_type): 
			self.noise_type=noise_type
			self.noise_mix_type=noise_mix_type
			
			self.noise_dim=attention_dim//2
			self.noise_mlp=nn.Linear(attention_dim, attention_dim//2)

		if not (encoder_dim==decoder_dim):
			self.decoder_h = nn.Sequential(nn.Linear(encoder_dim, decoder_dim), nn.Tanh())
	
		self.out = nn.Sequential(nn.Linear(decoder_dim, 2), nn.ReLU())

		self.att2enc=nn.Linear(attention_dim, encoder_dim)#, nn.ReLU())
		self.enc2att=nn.Linear(encoder_dim, attention_dim)#, nn.ReLU())
		self.att2dec = nn.Linear(attention_dim, decoder_dim)#, nn.ReLU())
		self.dec2att =nn.Linear(decoder_dim, attention_dim)
		
		if ('scene' in self.model_type):
			self.scene_attention=scene_attention_(pretrained_scene, attention_dim) 
			self.scene2att=nn.Linear(2*attention_dim, attention_dim)

	def init_states(self, total_peds, hidden_dim):
		h_t = Variable(torch.zeros(total_peds, hidden_dim).to(self.device), requires_grad=True)
		c_t = Variable(torch.zeros(1, total_peds, hidden_dim).to(self.device), requires_grad=True)
		return h_t, c_t
	
	def encode(self, x, dmat, bmat, hmat, mask, mean, var, scene_embedding=None, domain=None):
		batch_size, num_pedestrians = x.size()[:2]
		total_peds = batch_size*num_pedestrians
		h_t, c_t = self.init_states(total_peds, self.encoder_dim) 
		x = x.view(total_peds, -1, 2) 
		embedded_x = self.encoder_embedding(x) # total_peds x obs_len x embedding_dim 
		encoded_input=[]
		encoded_outputs=[]
		for i, x_i in enumerate(embedded_x.chunk(x.size(1), dim=1)):
			mask_i = mask[:,:,i].view(total_peds)
			_ , (h_t, c_t) = self.encoder(x_i, (h_t.unsqueeze(0), c_t))
			h_t = h_t.squeeze(0) # total_peds x encoder_dim
			if hasattr(self, 'spatial_attention'):
				if hasattr(self, 'enc2att'): h_t = self.enc2att(h_t) # total_peds x attention_dim
				h_t = h_t.view(batch_size, num_pedestrians, self.attention_dim)
				h_t = self.spatial_attention(h_t) #, dmat[:,:,i,:], bmat[:,:,i,:], hmat[:,:,i,:], mask[:,:,i], domain=domain)
				if not hasattr(self, 'scene_attention') and hasattr(self, 'att2enc'):
					h_t = self.att2enc(h_t)
			if hasattr(self, 'scene_attention'): # and not scene embedding is None:
				h_t = torch.cat((h_t, scene_embedding), dim=-1)
				#h_t = h_t*scene_embedding
				h_t = self.scene2att(h_t) # total_peds x attention_dim
				if hasattr(self, 'att2enc'): h_t = self.att2enc(h_t) # total_peds x encoder_dim
			encoded_input+=[h_t.view(batch_size, num_pedestrians, self.encoder_dim)]
		return h_t, encoded_input, encoded_outputs
	def decode(self, x, dmat, bmat, hmat, input_mask, output_mask, h_t, encoded_input, mean, var, scene_embedding=None, domain=None):
		if self.training: eps=1e-14
		else: eps=0
		# mean-> batch_size x 1 x feature_dim
		mean, var = mean.squeeze(1), var.squeeze(1) # batch_size x feature_dim
		batch_size, num_pedestrians = dmat.size()[:2]
		total_peds = batch_size*num_pedestrians
		_, c_t = self.init_states(total_peds, self.decoder_dim) 
		encoded_input=torch.stack(encoded_input, dim=2) # batch_size x num_pedestrians x obs_len x encoder_dim
		encoded_input=encoded_input.view(total_peds, self.obs_len, self.encoder_dim) # total_peds x obs_len x encoder_dim
		if hasattr(self, 'decoder_h'): h_t = self.decoder_h(h_t) # batch_size x num_pedestrians x obs_len x decoder_dim
		prediction=[]
		input_mask = input_mask.view(total_peds, self.obs_len)
		mask=output_mask.view(total_peds,self.pred_len)
		x_ = revert_orig_tensor(x.view(batch_size, num_pedestrians, 2), mean, var, input_mask[:,-1].view(batch_size, num_pedestrians), dim=1)
		distance_matrix, bearing_matrix, heading_matrix = dmat[:,:,self.obs_len-1,:], bmat[:,:,self.obs_len-1,:], hmat[:,:,self.obs_len,:]
		for j in range(self.pred_len):
			x = x.view(total_peds, 2) # batch_size x num_pedestrians x feature_dim -> total_peds x feature_dim  
			embedded_x=self.decoder_embedding(x) # total_peds x embedding_dim 
			_, (h_t,c_t)=self.decoder(embedded_x.unsqueeze(1),(h_t.unsqueeze(0),c_t))
			h_t = h_t.squeeze(0) # total_peds x decoder_dim 
			if hasattr(self, 'spatial_attention'):
				if hasattr(self, 'dec2att'): h_t=self.dec2att(h_t) # total_peds x attention_dim 
				h_t = h_t.view(batch_size, num_pedestrians, self.attention_dim) # batch_size x num_pedestrians x attention_dim 
				h_t=self.spatial_attention(h_t) #,distance_matrix,bearing_matrix,heading_matrix,mask[:,j].view(batch_size,num_pedestrians),domain=domain)
				if not hasattr(self, 'scene_attention') and hasattr(self, 'att2dec'): h_t=self.att2dec(h_t) # total_peds x decoder_dim
			if hasattr(self, 'scene_attention'): # and not scene embedding is None:
				#h_t = h_t*scene_embedding
				h_t = torch.cat((h_t, scene_embedding), dim=-1)
				h_t = self.scene2att(h_t) # total_peds x attention_dim
				if hasattr(self, 'att2dec'): h_t = self.att2dec(h_t) 
			if hasattr(self, 'temporal_attention'):
				h_t, scores=self.temporal_attention(h_t, encoded_input, input_mask) # input_mask -> batch_size x num_pedestrians x obs_len
			x_out=self.out(h_t) # total_peds x feature_dim 
			x_out=x_out*mask[:,j].unsqueeze(-1).expand_as(x_out)
			x_out_ = revert_orig_tensor(x_out.view(batch_size,num_pedestrians,2), mean, var, mask[:,j].view(batch_size, num_pedestrians), dim=1) # batch_size x num_pedestrians x feature_dim
			x_ = x_out_
			distance_matrix, bearing_matrix, heading_matrix=get_features(x_out_, 1, x_, mask=mask[:,j].view(batch_size, num_pedestrians), eps=eps) 
			x = x_out
			prediction+=[x_out.view(batch_size,num_pedestrians,2)] 
		prediction = torch.stack(prediction, dim=2)
		return prediction
	def forward(self, x, pedestrians, dmat, bmat, hmat, input_mask, output_mask, scene, mean=None, var=None, domain=None):
		# x -> batch_size x num_pedestrians x obs_len x feature_dim
		# dmat, bmat, hmat -> batch_size x num_pedestrians x obs_len x num_pedestrians
		# input_mask -> batch_size x num_pedestrians x obs_len
		# output_mask -> batch_size x num_pedestrians x pred_len 
		batch_size, num_pedestrians = x.size()[:2]
		total_peds = batch_size*num_pedestrians
		if hasattr(self, 'scene_attention'):
			scene_embedding = self.scene_attention(scene) # batch_size x attention_dim
			scene_embedding = scene_embedding.unsqueeze(1).expand(batch_size, num_pedestrians, self.attention_dim)
			scene_embedding = scene_embedding.contiguous().view(total_peds, self.attention_dim) 
		else:
			scene_embedding = None
		total_peds = batch_size*num_pedestrians
		final_h, encoded_input, encoded_outputs=self.encode(x, dmat, bmat, hmat, input_mask, mean, var, scene_embedding, domain=domain)
		if ('generative' in self.model_type):
			final_h = final_h.view(batch_size, num_pedestrians, self.encoder_dim)
			final_h = self.add_noise(final_h) 
		prediction=self.decode(x[:,:,-1,:].view(total_peds,2), dmat, bmat, hmat, input_mask, output_mask, final_h, encoded_input, mean, var, scene_embedding, domain=domain)
		return prediction, encoded_outputs
	def add_noise(self, h_t):
		batch_size, num_pedestrians, h_dim = h_t.size()
		total_peds=batch_size*num_pedestrians
		if hasattr(self, 'noise_mlp'): h_t=self.noise_mlp(h_t)
		if self.noise_mix_type=='ped':
			z = get_noise((batch_size, 1, self.noise_dim), self.noise_type, self.device)
			z = z.repeat(1, num_pedestrians, 1)
		elif self.noise_mix_type=='sample':
			z = get_noise((batch_size, num_pedestrians, self.noise_dim), self.noise_type, self.device)
		h_t = torch.cat((h_t, z), dim=2)
		h_t = h_t.view(total_peds, h_dim)
		return h_t 

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
			self.encoder=nn.LSTM(embedding_dim, hidden_size, num_layers=self.num_layers, batch_first=True)
		elif model_type=='fc':
			self.fc = nn.Sequential(nn.Linear(self.embedding_dim*self.seq_len, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, hidden_size), nn.LeakyReLU())
		self.embedding=nn.Sequential(nn.Linear(feature_dim, embedding_dim, bias=False), nn.ReLU())
		self.encoder_spatial_embedding=nn.Linear(2*attention_dim, attention_dim)


		self.spatial_attention=spatial_attention(delta_bearing, delta_heading, domain_parameter, attention_dim)
		
		self.classifier = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, 1))

		self.enc2att=nn.Linear(hidden_size, attention_dim)
		self.att2enc=nn.Linear(attention_dim, hidden_size)

		self.sigmoid=nn.Sigmoid()
	def init_hidden(self, size_x):
		return (Variable(torch.zeros(self.num_layers, size_x[0], self.hidden_size), requires_grad=True).cuda(), 
			Variable(torch.zeros(self.num_layers, size_x[0], self.hidden_size), requires_grad=True).cuda()
			)
	def forward(self, x, dmat=None, bmat=None, hmat=None, mask=None, domain=None):
		x = self.embedding(x)
		batch_size, num_pedestrians = x.size()[:2]
		total_peds=batch_size*num_pedestrians
		x = x.view(total_peds, -1, self.embedding_dim)
		h_t, c_t = self.init_hidden(x.size())
		if not dmat is None:
			for i in range(self.seq_len):
				x_i = x[:,i,:]
				x_i = x_i.view(total_peds, 1, self.embedding_dim)
				_, (h_t, c_t) = self.encoder(x_i, (h_t.view(1, total_peds, self.hidden_size), c_t))
				if hasattr(self, 'spatial_attention'):
					if hasattr(self, 'enc2att'):
						h_t = self.enc2att(h_t)
					h_t = h_t.view(batch_size, num_pedestrians, self.hidden_size)
					h_t=self.spatial_attention(h_t,dmat[:,:,i,:],bmat[:,:,i,:],hmat[:,:,i,:],mask[:,:,i].view(batch_size,num_pedestrians), domain)
					h_t = self.encoder_spatial_embedding(h_t)
					if hasattr(self, 'att2enc'):
						h_t = self.att2enc(h_t)
		else:
			if hasattr(self, 'encoder'):
				output, (h_t, c_t) = self.encoder(x, (h_t, c_t))
			elif hasattr(self, 'fc'):
				h_t = self.fc(x.view(batch_size, num_pedestrians, self.embedding_dim*self.seq_len))
		scores = self.classifier(h_t.view(batch_size, num_pedestrians, self.hidden_size))
		scores = self.sigmoid(scores)
		return scores 


