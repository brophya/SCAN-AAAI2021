import torch
import random
import time

from utils import *

def bce_loss(pred, gt, num_peds):
	neg_abs = -pred.abs()
	loss = pred.clamp(min=0) - pred * gt + (1 + neg_abs.exp()).log()
	loss = loss.sum()/num_peds.sum()
	return loss

def gan_g_loss(scores_fake):
	y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2) 
	return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
	y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
	y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
	loss_real = bce_loss(scores_real, y_real)
	loss_fake = bce_loss(scores_fake, y_fake)
	return loss_real + loss_fake

def traj_similarity(predictions, mask, eps=1e-16, k=1):
	best_k, batch_size, num_pedestrians, prediction_length, feature_size = list(predictions.size())
	num_traj = (batch_size*num_pedestrians)
	predictions = predictions.view(best_k, num_traj, prediction_length, feature_size)
	predictions = predictions.transpose(0,1)
	mask = mask.view(-1, prediction_length)
	similarity_score=[]
	for t in range(num_traj):
		mask_t = mask[t,:] # prediction_length
		mask_t = mask_t.unsqueeze(-1).unsqueeze(-1).expand(prediction_length, best_k, best_k)
		mask_t = mask_t*mask_t.transpose(1,2)
		pred = predictions[t,...] # 1 x num_traj x prediction_length x feature_size
		dist = get_distance_matrix(pred.transpose(0,1), neighbors_dim=1) 
		s = torch.exp(-k*dist+eps) 
		s = s*mask_t
		s[:, range(best_k), range(best_k)].data.fill_(0.0)
		s = s.sum()
		s = s.div(prediction_length*best_k) 
		similarity_score.append(s)
	similarity = torch.stack(similarity_score, dim=0).mean()
	return similarity

