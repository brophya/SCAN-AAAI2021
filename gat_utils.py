from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
	def __init__(self, ip_dim, op_dim):
		super(GraphAttentionLayer, self).__init__()
		self.ip_dim = ip_dim
		self.op_dim = ip_dim
		#self.w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(ip_dim, op_dim).float()))
		self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(ip_dim, 1).float()))
		self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(ip_dim, 1).float()))
		self.leakyrelu = nn.LeakyReLU(0.1)
	def forward(self, ip):
		batch_size, num_peds, ip_dim = list(ip.size())
		adj = torch.ones(batch_size, num_peds, num_peds).to(ip.device)
		#h = torch.bmm(ip, self.w.expand(batch_size, ip_dim, -1))
		f_1 = torch.bmm(ip, self.a1.expand(batch_size, self.op_dim, 1))
		f_2 = torch.bmm(ip, self.a2.expand(batch_size, self.op_dim, 1))
		e = self.leakyrelu(f_1 + f_2.transpose(2,1))
		zero_vec = -9e15*torch.ones_like(e)
		attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=1)
		h_prime = torch.bmm(attention, ip)
		h_prime = F.elu(h_prime)
		return h_prime 
