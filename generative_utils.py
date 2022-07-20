from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
from losses import traj_similarity

adv_loss = nn.BCELoss()

def discriminator_step(b, batch, generator, discriminator, optimizer_d, d_spatial=False, eps=1e-06, d_type='global', d_domain=False):
	discriminator.train()
	optimizer_d.zero_grad()

	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, op_mask, pedestrians, batch_mean, batch_var = batch
	prediction = generator(sequence, pedestrians, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask, scene = None, mean = batch_mean, var = batch_var) 
	prediction = revert_orig_tensor(prediction, batch_mean, batch_var, op_mask, dim=1)    
	target = revert_orig_tensor(target, batch_mean, batch_var, op_mask, dim=1)

	obs_len = sequence.size(2) 
	if d_spatial:
		pred_dmat, pred_bmat, pred_hmat = get_features(prediction,1) 
		pred_bmat, pred_hmat, pred_dmat = mask_matrix(pred_bmat,op_mask,(1,3)), mask_matrix(pred_hmat,op_mask,(1,3)), mask_matrix(pred_dmat,op_mask,(1,3))
	
		target_dmat, target_bmat, target_hmat = get_features(target,1)
		target_dmat, target_bmat, target_hmat = mask_matrix(target_dmat, op_mask, (1,3)), mask_matrix(target_bmat, op_mask, (1,3)), mask_matrix(target_hmat, op_mask, (1,3))
	prediction = normalize_tensor(prediction, batch_mean, batch_var, op_mask, dim=1)
	target = normalize_tensor(target, batch_mean, batch_var, op_mask, dim=1)
	
	domain=None
	if not d_domain and hasattr(generator, 'spatial_attention'):
		domain = generator.spatial_attention.domain

	obs_len=sequence.size(2)
	
	
	if d_type=='global':
		prediction = torch.cat((sequence, prediction), dim=2)
		op_mask = torch.cat((ip_mask, op_mask), dim=-1) 
		target = torch.cat((sequence, target), dim=2)	

	if d_spatial:
		
		if d_type=='global':
			pred_dmat = torch.cat((dist_matrix[:,:,:obs_len,:], pred_dmat), dim=2)
			pred_bmat = torch.cat((bearing_matrix[:,:,:obs_len,:], pred_bmat), dim=2)
			pred_hmat = torch.cat((heading_matrix[:,:,:obs_len,:], pred_hmat), dim=2)
		
		scores_fake = discriminator(prediction, pred_dmat, pred_bmat, pred_hmat, op_mask, domain=domain) 
	else:
		scores_fake = discriminator(prediction)
		
	scores_fake = scores_fake.view(-1)[~(op_mask[...,-1].view(-1)==0)]
	fake = Variable(torch.zeros_like(scores_fake), requires_grad=False)
	fake_loss = adv_loss(scores_fake, fake) 
	if d_spatial:
		obs_len=sequence.size(2)
		if d_type=='global':
			dist_matrix = torch.cat((dist_matrix, target_dmat), 2)
			bearing_matrix=torch.cat((bearing_matrix, target_bmat),2)
			heading_matrix=torch.cat((heading_matrix,target_hmat),2)
			
			scores_real = discriminator(target, dist_matrix, bearing_matrix, heading_matrix, op_mask, domain=domain)
		else:
			
			scores_real = discriminator(target, target_dmat, target_bmat, target_hmat, op_mask, domain=domain)

	else:
		scores_real = discriminator(target)
	
	scores_real = scores_real.view(-1)[~(op_mask[...,-1].view(-1)==0)]
	valid = Variable(torch.ones_like(scores_real), requires_grad=False)
	real_loss = adv_loss(scores_real, valid)

	loss = real_loss+fake_loss
	
	loss.backward()
	optimizer_d.step()
	return loss

def generator_step(b, batch, generator, discriminator=None, optimizer_g=None, best_k=None, l=None, train=True, d_spatial=False, l2_loss_weight=1, clip=None, d_type='global',d_domain=False):
	eps=0
	reduction='sum'
	if generator.training: 
		optimizer_g.zero_grad()
		eps=1e-14
		reduction='mean'
	min_ade=float(np.inf) 
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, op_mask, pedestrians, batch_mean, batch_var = batch
	batch_size=sequence.size(0)
	target_mask = op_mask.unsqueeze(-1).expand(target.size())
	predictions = []
	target = revert_orig_tensor(target, batch_mean, batch_var, op_mask, dim=1)  
	for k in range(best_k):
		prediction = generator(sequence, pedestrians, dist_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask, scene = None, mean = batch_mean, var = batch_var) 
		prediction = revert_orig_tensor(prediction, batch_mean, batch_var, op_mask, dim=1) 
		ade_g, fde_g = eval_metrics(prediction, target, pedestrians, op_mask,eps=eps, reduction=reduction)
		predictions+=[prediction]
		if (k==0):
			first_ade=ade_g
		if ade_g<min_ade:
			min_ade=ade_g
			fde_ = fde_g 
	final_pred = prediction
	predictions = torch.stack(predictions, dim=0) 
	if generator.training:
		if not (l==0):
			similarity_metric = traj_similarity(predictions, op_mask)
		
		if d_spatial:
			pred_dmat, pred_bmat, pred_hmat = get_features(final_pred,1)
			pred_bmat, pred_hmat, pred_dmat = mask_matrix(pred_bmat,op_mask,(1,3)), mask_matrix(pred_hmat,op_mask,(1,3)), mask_matrix(pred_dmat,op_mask,(1,3))
		final_pred = normalize_tensor(final_pred, batch_mean, batch_var, op_mask, dim=1)

		if d_type=='global':
			final_pred = torch.cat((sequence, final_pred), dim=2)
		
		if d_spatial:
			obs_len = sequence.size(2)
			if d_type=='global':
				pred_dmat = torch.cat((dist_matrix[:,:,:obs_len,:], pred_dmat), dim=2)
				pred_bmat = torch.cat((bearing_matrix[:,:,:obs_len,:], pred_bmat), dim=2)
				pred_hmat = torch.cat((heading_matrix[:,:,:obs_len,:], pred_hmat), dim=2)
				op_mask = torch.cat((ip_mask, op_mask), dim=-1)

			domain=None
			if not d_domain and hasattr(generator, 'spatial_attention'):
				domain = generator.spatial_attention.domain
			

			scores_fake = discriminator(final_pred, pred_dmat, pred_bmat, pred_hmat, op_mask, domain=domain)
		else:
			scores_fake = discriminator(final_pred)
		scores_fake = scores_fake.view(-1)[~(op_mask[...,-1].view(-1)==0)]
		valid = Variable(torch.ones_like(scores_fake), requires_grad=False)
		discriminator_loss = adv_loss(scores_fake, valid) 

		loss = min_ade+discriminator_loss
		if not (l==0):
			loss = loss+l*similarity_metric 
		loss.backward()
		if not clip is None:
			nn.utils.clip_grad_norm_(generator.parameters(), clip)
		optimizer_g.step()
	
	if not train: 
		discriminator_loss=None
	
	return discriminator_loss,  min_ade, fde_, final_pred, pedestrians, first_ade

def check_accuracy(loader, generator, discriminator, num_traj=1):
	generator.eval()
	discriminator.eval()
	mon_ade = float(0)
	first_ade = float(0)
	test_fde = float(0)
	d_loss = float(0)
	total_peds=float(0)
	with torch.no_grad():
		for b, batch in enumerate(loader):
			_, min_ade, fde, final_pred, pedestrians, ade = generator_step(b, batch, generator, discriminator=discriminator, best_k=num_traj, train=False)
			total_peds+=pedestrians.sum()
			mon_ade+=min_ade
			first_ade+=ade
			test_fde+=fde
		mon_ade/=(total_peds*final_pred.size(2))
		test_fde/=(total_peds)
		first_ade/=(total_peds*final_pred.size(2))
	return mon_ade, test_fde, first_ade
