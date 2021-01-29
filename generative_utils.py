from __future__ import print_function

import torch
import torch.nn as nn

from model import *
from utils import *
from losses import *

adv_loss = bce_loss #nn.BCELoss() #

def get_distance_matrix(sample,neighbors_dim=0,eps=1e-24):
	if not neighbors_dim==1:
		sample=sample.transpose(0,1)
	n = sample.size(1)
	s = sample.size(0)
	if len(sample.size())==3:
		norms=torch.sum((sample)**2,dim=2,keepdim=True)
		norms = norms.expand(s, n, n) + norms.expand(s, n, n).transpose(1,2)
		dsquared=norms-2*torch.bmm(sample,sample.transpose(1,2))
	else:
		norms=torch.sum((sample)**2,dim=3,keepdim=True)
		norms = norms.transpose(1,2) # batch_size x prediction_length x num_pedestrians x 1
		norms = norms.expand(s, sample.size(2), n, n) + norms.expand(s, sample.size(2), n, n).transpose(2,3)
		dsquared=norms-2*torch.matmul(sample.transpose(1,2),sample.transpose(1,2).transpose(2,3))
	distance = torch.sqrt(torch.abs(dsquared)+eps)
	if not neighbors_dim==1:
		distance=distance.transpose(0,1)
	if len(sample.size())==4:
		distance = distance.transpose(1,2)	
	return distance

def get_features(sample,neighbors_dim,previous_sequence=None):
	offset_bearing, offset_heading = 0,0
	if not (offset_heading==0 and offset_bearing==0):
		warnings.warn("all computation for features with offsets: " + str(offset_bearing) + " " + str(offset_heading))
	if not neighbors_dim==1:	
		sample=sample.transpose(0,1)
	n = sample.size(1)
	s = sample.size(0)
	if len(sample.size())==3:
		shape=(s, n, n)
	else:
		shape = (s, n, sample.size(2), n)
	x1 = sample[...,0]
	y1 = sample[...,1]
	x1 = x1.unsqueeze(-1).expand(shape)
	y1 = y1.unsqueeze(-1).expand(shape)
	if len(sample.size())==3:
		x2 = x1.transpose(1,2)
		y2 = y1.transpose(1,2)
	else:
		x2 = x1.transpose(1,3)
		y2 = y1.transpose(1,3) 
	dx = x2-x1
	dy = y2-y1
	bearing = rad2deg * (torch.atan2(dy,dx))
	bearing = torch.where(bearing<0, bearing+360, bearing)
	distance = get_distance_matrix(sample,1)
	heading = get_heading(sample,prev_sample=previous_sequence)
	if len(sample.size())==3:
		heading = heading.unsqueeze(-1).expand(shape)
	else:
		heading = heading.unsqueeze(-1).expand(shape)
	bearing=bearing-heading
	bearing = torch.where(distance.data==distance.data.min(), torch.zeros_like(bearing), bearing)
	if len(sample.size())==3:
		heading = heading.transpose(1,2)-heading
	else:
		heading = heading.transpose(1, 3) - heading
	bearing = torch.where(bearing<0, bearing+360, bearing)
	heading = torch.where(heading<0, heading+360, heading)
	if not neighbors_dim==1:
		distance, bearing, heading = distance.transpose(0,1),bearing.transpose(0,1),heading.transpose(0,1)
	return distance, bearing, heading

def get_heading(sample,prev_sample=None):
	diff = torch.zeros_like(sample)
	if prev_sample is None:
		if len(sample.size())==3:
			diff[1:,...] = sample[1:,...]-sample[:-1,...]
			diff[0,...] = diff[1,...] 
			heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
			heading = torch.where(heading<0, heading+360, heading)
		else:
			diff[:,:,1:,...] = sample[:,:,1:,...] - sample[:,:,-1:,...]
			diff[:,:,0,...] = diff[:,:,1,...]
			heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
			heading = torch.where(heading<0, heading+360, heading)
	else:
		diff = sample-prev_sample
		heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
		heading = torch.where(heading<0, heading+360, heading)
	return heading

def discriminator_step(b, batch, generator, discriminator, optimizer_d, eps=1e-06):
	discriminator.zero_grad()
	if hasattr(discriminator, 'spatial_attn') and discriminator.spatial_attn.domain.requires_grad is False:  discriminator.spatial_attn.domain.data.copy_(generator.spatial_attention.domain.data)	
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians, scene_context, batch_mean, batch_var, frame_id = batch
	out_g, target, sequence, ped_count, distance_matrix, bearing_matrix, heading_matrix = predict(batch, generator, num_traj=1)
	out_g=out_g[0]
	ade_g = ade(out_g, target, ped_count)
	fde_g = fde(out_g, target, ped_count)
	target_d, target_b, target_h = get_features(out_g, 1)
	target_d_real, target_b_real, target_h_real = get_features(target, 1)
	#batch_mean, batch_var = batch_mean.squeeze(1), batch_var.squeeze(1)
	batch_mean, batch_var = batch_mean.unsqueeze(1).expand_as(out_g), batch_var.unsqueeze(1).expand_as(out_g)
	out_g = out_g-batch_mean
	out_g = out_g/batch_var
	target = target-batch_mean
	target =target/batch_var
	input_d = torch.cat([sequence, out_g], dim=2)
	input_d_real = torch.cat([sequence, target], dim=2)
	dmat, bmat, hmat = torch.cat([distance_matrix, target_d], dim=2), torch.cat([bearing_matrix, target_b], dim=2), torch.cat([heading_matrix, target_h], dim=2)
	dmat_real, bmat_real, hmat_real = torch.cat([distance_matrix, target_d_real], dim=2), torch.cat([bearing_matrix, target_b_real], dim=2), torch.cat([heading_matrix, target_h_real], dim=2)
	#scores_fake = discriminator(out_g, target_d, target_b, target_h, ip_mask, op_mask)
	scores_fake = discriminator(input_d, dmat, bmat, hmat, ip_mask, op_mask)
	fake = Variable(torch.zeros_like(scores_fake), requires_grad=False).uniform_(0.0, 0.3)
	fake_loss = adv_loss(scores_fake+eps, fake)
	#scores_real = discriminator(target, target_d_real, target_b_real, target_h_real, ip_mask, op_mask)
	scores_real = discriminator(input_d_real, dmat_real, bmat_real, hmat_real, ip_mask, op_mask)
	valid = Variable(torch.ones_like(scores_real), requires_grad=False).uniform_(0.7, 1.0)
	real_loss = adv_loss(scores_real+eps, valid)
	loss = real_loss+fake_loss
	#loss = loss/2
	total_loss = loss+ade_g
	total_loss.backward()
	#for param in list(discriminator.named_parameters()): print(f"{param[0]}: {param[1].grad.max()}")
	#for param in list(discriminator.named_parameters()): 
	#	if (not 'domain' in param[0]) and (param[1].grad.max())==0: print(f"[D]\t{param[0]} has zero grad!") #assert(not(param[1].grad.max()==0)), f"{param[0]} has zero grad!"
	optimizer_d.step()
	return loss

def generator_step(b, batch, generator, discriminator=None, optimizer_g=None, best_k=None, weight_sim=None, train=True):
	if train: 
		generator.zero_grad()
		if hasattr(discriminator, 'spatial_attn') and discriminator.spatial_attn.domain.requires_grad is False:  discriminator.spatial_attn.domain.data.copy_(generator.spatial_attention.domain.data)  
	min_ade=Variable(torch.FloatTensor(1), requires_grad=True).to(device).fill_(1000) 
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians, scene_context, batch_mean, batch_var, frame_id = batch
	out_g, target, sequence, ped_count, distance_matrix, bearing_matrix, heading_matrix = predict(batch, generator, num_traj=best_k)
	for k in range(best_k):
		ade_g = ade(out_g[k], target, ped_count)
		fde_g = fde(out_g[k], target, ped_count)
		if ade_g.item()<min_ade.item():
			min_ade=ade_g
			fde_ = fde_g 
			final_pred = out_g[k]
	if train and not (weight_sim==0):
		predictions = torch.stack(out_g, dim=0)
		similarity_metric = traj_similarity(predictions, op_mask)
	if train:
		target_d, target_b, target_h = get_features(final_pred, 1)
		batch_mean, batch_var = batch_mean.unsqueeze(1).expand_as(final_pred), batch_var.unsqueeze(1).expand_as(final_pred)	
		final_pred = final_pred-batch_mean
		final_pred = final_pred/batch_var
		input_d = torch.cat([sequence, final_pred], dim=2)
		dmat, bmat, hmat = torch.cat([distance_matrix, target_d], dim=2), torch.cat([bearing_matrix, target_b], dim=2), torch.cat([heading_matrix, target_h], dim=2)
		scores_fake = discriminator(input_d, dmat, bmat, hmat, ip_mask, op_mask)
		valid = Variable(torch.ones_like(scores_fake), requires_grad=False).uniform_(0.0, 0.3)
		discriminator_loss = adv_loss(scores_fake, valid)
		loss = min_ade+discriminator_loss
		if not (weight_sim==0):
			loss = loss+weight_sim*similarity_metric 
		loss.backward()
		#for param in list(discriminator.named_parameters()): 
		#	if (not 'domain' in param[0]) and param[1].grad.max()==0: print(f"[G]\t{param[0]} has zero grad!") #assert(not(param[1].grad.max()==0)), f"{param[0]} has zero grad!"
		optimizer_g.step()
	if not train: discriminator_loss=None
	return discriminator_loss,  min_ade, fde_

def check_accuracy(loader, generator, discriminator, plot_traj=False, num_traj=1):
	generator.eval()
	test_ade = float(0)
	test_fde = float(0)
	d_loss = float(0)
	with torch.no_grad():
		for b, batch in enumerate(loader):
			_, min_ade, fde = generator_step(b, batch, generator, best_k=num_traj, train=False)
			test_ade+=min_ade
			test_fde+=fde
		test_ade/=(b+1)
		test_fde/=(b+1)
	return test_ade, test_fde
