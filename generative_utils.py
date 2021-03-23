from __future__ import print_function

import torch
import torch.nn as nn

from model import *
from utils import *
from losses import *

adv_loss = nn.BCELoss()

def discriminator_step(b, batch, generator, discriminator, optimizer_d, d_spatial=False, eps=1e-06, d_type='global', d_domain=False):
	optimizer_d.zero_grad()

	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, op_mask, pedestrians, scene_context, batch_mean, batch_var = batch
	prediction, target, sequence, pedestrians, op_mask, _= predict(batch, generator)
	
	obs_len = sequence.size(2) 
	if d_spatial:
		pred_dmat, pred_bmat, pred_hmat = get_features(prediction,1) 
		pred_bmat, pred_hmat, pred_dmat = mask_matrix(pred_bmat,op_mask,(1,3)), mask_matrix(pred_hmat,op_mask,(1,3)), mask_matrix(pred_dmat,op_mask,(1,3))
	sequence = normalize_tensor(sequence, batch_mean, batch_var, ip_mask, dim=1) 
	prediction = normalize_tensor(prediction, batch_mean, batch_var, op_mask, dim=1)
	target = normalize_tensor(target, batch_mean, batch_var, op_mask, dim=1)
	
	domain=None
	if not d_domain:
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
	fake = Variable(torch.zeros_like(scores_fake), requires_grad=False).uniform_(0.0, 0.3)
	fake_loss = adv_loss(scores_fake, fake) 
	if d_spatial:
		obs_len=sequence.size(2)
		if d_type=='global':
			scores_real = discriminator(target, dist_matrix, bearing_matrix, heading_matrix, op_mask, domain=domain)
		else:
			
			scores_real = discriminator(target, dist_matrix[:,:,obs_len:,:], bearing_matrix[:,:,obs_len:,:], heading_matrix[:,:,obs_len:,:], op_mask, domain=domain)

	else:
		scores_real = discriminator(target)
	
	
	scores_real = scores_real.view(-1)[~(op_mask[...,-1].view(-1)==0)]
	valid = Variable(torch.ones_like(scores_real), requires_grad=False).uniform_(0.8,1.0)
	real_loss = adv_loss(scores_real, valid)

	loss = real_loss+fake_loss
	loss = loss/2
	
	loss.backward()

	optimizer_d.step()
	return loss

def generator_step(b, batch, generator, discriminator=None, optimizer_g=None, best_k=None, l=None, train=True, d_spatial=False, l2_loss_weight=1, clip=None, d_type='global',d_domain=False):
	if generator.training: 
		optimizer_g.zero_grad()
	min_ade=float(1000)
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, op_mask, pedestrians, scene_context, batch_mean, batch_var = batch
	batch_size=sequence.size(0)
	predictions, target, sequence, pedestrians = predict_multiple(batch, generator, best_k)
	
	ade_vals = [] 
	for k in range(best_k):
		ade_g, fde_g = eval_metrics(predictions[k,...], target, pedestrians, op_mask)
		ade_vals+=[ade_g]
		if ade_g.item()<min_ade:
			min_ade=ade_g
			fde_ = fde_g 
			final_pred = predictions[k]
	if generator.training:
		if not (l==0):
			similarity_metric = traj_similarity(predictions, op_mask)
		
		if d_spatial:
			pred_dmat, pred_bmat, pred_hmat = get_features(final_pred,1)
			pred_bmat, pred_hmat, pred_dmat = mask_matrix(pred_bmat,op_mask,(1,3)), mask_matrix(pred_hmat,op_mask,(1,3)), mask_matrix(pred_dmat,op_mask,(1,3))
				
		final_pred = normalize_tensor(final_pred, batch_mean, batch_var, op_mask, dim=1)
		sequence = normalize_tensor(sequence, batch_mean, batch_var, ip_mask, dim=1)

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
			if not d_domain:
				domain = generator.spatial_attention.domain
			
			scores_fake = discriminator(final_pred, pred_dmat, pred_bmat, pred_hmat, op_mask, domain=domain)
		else:
			scores_fake = discriminator(final_pred)
		scores_fake = scores_fake.view(-1)[~(op_mask[...,-1].view(-1)==0)]
		valid = Variable(torch.ones_like(scores_fake), requires_grad=False).uniform_(0.8,1.0)
		discriminator_loss = adv_loss(scores_fake, valid) 

		loss = l2_loss_weight*min_ade+(1-l2_loss_weight)*discriminator_loss
		if not (l==0):
			loss = loss+l*similarity_metric 
		loss.backward()
		if not clip is None:
			nn.utils.clip_grad_norm_(generator.parameters(), clip)
		optimizer_g.step()
	
	if not train: 
		discriminator_loss=None
	
	return discriminator_loss,  min_ade, fde_, final_pred

def check_accuracy(loader, generator, discriminator, plot_traj=False, num_traj=1):
	generator.eval()
	discriminator.eval()
	test_ade = float(0)
	test_fde = float(0)
	d_loss = float(0)
	with torch.no_grad():
		for b, batch in enumerate(loader):
			_, min_ade, fde, _ = generator_step(b, batch, generator, discriminator=discriminator, best_k=num_traj, train=False)
			test_ade+=min_ade
			test_fde+=fde
		test_ade/=(b+1)
		test_fde/=(b+1)
	return test_ade, test_fde
