from __future__ import print_function

import torch
import random
import time

def bce_loss(pred, gt):
	neg_abs = -pred.abs()
	loss = pred.clamp(min=0) - pred * gt + (1 + neg_abs.exp()).log()
	return loss.mean()

def gan_g_loss(scores_fake):
	y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2) 
	return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
	y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
	y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
	loss_real = bce_loss(scores_real, y_real)
	print(f"loss_real: {loss_real}")
	loss_fake = bce_loss(scores_fake, y_fake)
	print(f"loss_fake: {loss_fake}")
	return loss_real + loss_fake

def traj_similarity(predictions, mask, eps=1e-16, k=1):
	# batch_size x num_pedestrians x prediction_length 
	# best_k x batch_size x num_pedestrians x prediction_length x feature_size
	# (1) best_k x (batch_size x num_pedestrians) x prediction_length x feature_size 
	best_k, batch_size, num_pedestrians, prediction_length, feature_size = list(predictions.size())
	mask = mask.repeat(best_k, 1 , 1).view(best_k, batch_size, num_pedestrians, prediction_length)
	num_traj = (batch_size*num_pedestrians)
	mask = mask.view(best_k, num_traj, prediction_length) 
	predictions = predictions.view(best_k, num_traj, prediction_length, feature_size)
	predictions = predictions.transpose(0,1)
	# predictions -> (num_traj, best_k, prediction_length, feature_size)
	# (batch_size x num_pedestrians x prediction_length x feature_size)
	norms = torch.sum((predictions)**2, dim=3, keepdim=True)
	norms = norms.transpose(1,2)
	# norms -> (batch_size x num_pedestrians x prediction_length x (1?))
	norms = norms.expand(num_traj, prediction_length, best_k, best_k) + norms.expand(num_traj, prediction_length, best_k, best_k).transpose(2,3)
	dsquared = norms - 2*torch.matmul(predictions.transpose(1,2), predictions.transpose(1,2).transpose(2,3))
	distance = torch.sqrt(torch.abs(dsquared)+eps)
	distance=15*distance
	distance = distance**2
	S = torch.exp(-k*distance+eps)
	# sum all non diagonal elements??
	S[:,:, range(best_k), range(best_k)].data.fill_(0.0)
	# num_traj x prediction_length x best_k x best_k
	mask = mask.transpose(0,1).transpose(1,2).repeat(1, 1, best_k).view_as(S)
	S.data.masked_fill_(mask=~mask, value=float(0))
	similarity_sum = S.sum()
	similarity_sum = similarity_sum.div(batch_size*num_pedestrians)
	return similarity_sum	

