from __future__ import print_function

import torch
import torch.nn as nn

from model import *
from metrics import *
from losses import *

adv_loss = nn.BCELoss()

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
	distance = 15*distance 
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
		heading = heading.repeat(1,n).view(shape)
	else:
		heading = heading.repeat(1, 1, n).view(shape)
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
	optimizer_d.zero_grad()
	batch = [tensor.to(device) for tensor in batch]
	sequence, target, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask, ped_count = batch
	sequence, target, distance_matrix, bearing_matrix, heading_matrix = sequence.float(), target.float(), distance_matrix.float(), bearing_matrix.float(), heading_matrix.float()
	ip_mask, op_mask = ip_mask.bool(), op_mask.bool()
	out_g = generator(sequence, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask)
	op_mask_ = op_mask.unsqueeze(-1).expand_as(out_g)
	out_g.data.masked_fill_(mask=~op_mask_, value=float(0))
	target.data.masked_fill_(mask=~op_mask_, value=float(0))
	ade_g = ade(out_g, target, ped_count)
	fde_g = fde(out_g, target, ped_count)
	input_d = torch.cat([sequence, out_g], dim=2)
	input_d_real = torch.cat([sequence, target], dim=2)
	target_d, target_b, target_h = get_features(out_g, 1)
	target_d_real, target_b_real, target_h_real = get_features(target, 1)
	dmat, bmat, hmat = torch.cat([distance_matrix, target_d], dim=2), torch.cat([bearing_matrix, target_b], dim=2), torch.cat([heading_matrix, target_h], dim=2)
	dmat_real, bmat_real, hmat_real = torch.cat([distance_matrix, target_d_real], dim=2), torch.cat([bearing_matrix, target_b_real], dim=2), torch.cat([heading_matrix, target_h_real], dim=2)
	scores_fake = discriminator(input_d, dmat, bmat, hmat, ip_mask, op_mask)
	scores_real = discriminator(input_d_real, dmat_real, bmat_real, hmat_real, ip_mask, op_mask)
	assert(not(torch.isnan(scores_real).any()))
	assert(not(torch.isnan(scores_fake).any()))
	valid = Variable(torch.ones_like(scores_real), requires_grad=False).uniform_(0.7, 1.0)
	fake = Variable(torch.zeros_like(scores_real), requires_grad=False).uniform_(0.0, 0.3) 
	real_loss = adv_loss(scores_real+eps, valid)
	fake_loss = adv_loss(scores_fake+eps, fake)
	loss = (real_loss+fake_loss)/2 
	assert(not(torch.isnan(loss).any()))
	assert(not(torch.isnan(ade_g).any()))
	total_loss=loss+ade_g
	assert(not(torch.isnan(total_loss).any()))
	total_loss.backward()
	optimizer_d.step()
	return loss

def generator_step(b, batch, generator, discriminator, optimizer_g, best_k, weight_sim):
	optimizer_g.zero_grad()
	batch = [tensor.to(device) for tensor in batch]
	sequence, target, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask, ped_count = batch
	sequence, target, distance_matrix, bearing_matrix, heading_matrix = sequence.float(), target.float(), distance_matrix.float(), bearing_matrix.float(), heading_matrix.float()
	ip_mask, op_mask = ip_mask.bool(), op_mask.bool()
	min_ade=Variable(torch.FloatTensor(1), requires_grad=True).to(device).fill_(1000) 
	predictions = []
	for k in range(best_k):
		out_g = generator(sequence, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask)
		op_mask_ = op_mask.unsqueeze(-1).expand_as(out_g)
		out_g.data.masked_fill_(mask=~op_mask_, value=float(0))
		predictions+=[out_g]
		target.data.masked_fill_(mask=~op_mask_, value=float(0))
		ade_g = ade(out_g, target, ped_count)
		fde_g = fde(out_g, target, ped_count)
		if ade_g.item()<min_ade.item():
			min_ade=ade_g
			fde_ = fde_g 
			final_pred = out_g
	if not (weight_sim==0):
		predictions = torch.stack(predictions, dim=0)
		similarity_metric = traj_similarity(predictions, op_mask)
	input_d = torch.cat([sequence, final_pred], dim=2)
	target_d, target_b, target_h = get_features(final_pred, 1)
	dmat, bmat, hmat = torch.cat([distance_matrix, target_d], dim=2), torch.cat([bearing_matrix, target_b], dim=2), torch.cat([heading_matrix, target_h], dim=2)
	scores_fake = discriminator(input_d, dmat, bmat, hmat, ip_mask, op_mask)
	valid = Variable(torch.ones_like(scores_fake), requires_grad=False).uniform_(0.0, 0.3)
	discriminator_loss = adv_loss(scores_fake, valid)
	loss = min_ade+discriminator_loss
	if not (weight_sim==0):
		loss = loss+weight_sim*similarity_metric 
	loss.backward()
	optimizer_g.step()
	return discriminator_loss,  min_ade, fde_

def check_accuracy(loader, generator, discriminator, plot_traj=False, num_traj=1):
	generator.eval()
	test_ade = float(0)
	test_fde = float(0)
	d_loss = float(0)
	with torch.no_grad():
		for b, batch in enumerate(loader):
			batch = [tensor.to(device) for tensor in batch]
			sequence, target, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask, ped_count = batch
			sequence, target, distance_matrix, bearing_matrix, heading_matrix = sequence.float(), target.float(), distance_matrix.float(), bearing_matrix.float(), heading_matrix.float()
			ip_mask, op_mask = ip_mask.bool(), op_mask.bool()
			ade_g = []
			fde_g = []
			traj = []
			for i in range(num_traj):
				out_g = generator(sequence, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask)
				op_mask_ = op_mask.unsqueeze(-1).expand_as(out_g)
				out_g.data.masked_fill_(mask=~op_mask_, value=float(0))
				target.data.masked_fill_(mask=~op_mask_, value=float(0))
				traj+=[out_g]
				ade_g+=[ade(out_g, target, ped_count)]
				fde_g+=[fde(out_g, target, ped_count)]
			ade_b = min(ade_g)
			test_ade+=ade_b 
			for ix, ade_ in enumerate(ade_g):
				if ade_ == ade_b: fde_b=fde_g[ix]
			test_fde+=fde_b
		test_ade/=(b+1)
		test_fde/=(b+1)
	return test_ade, test_fde
