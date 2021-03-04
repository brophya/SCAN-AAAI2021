from __future__ import print_function

import torch
import torch.nn as nn

from model import *
from utils import *
from losses import *

adv_loss = nn.BCELoss() #

def batched_features(pred):
	dmat, bmat, hmat = [], [], []
	for p, pred_b in enumerate(pred.chunk(pred.size(0), dim=0)):
		# for each batch
		b_dmat, b_bmat, b_hmat = get_features(pred_b.squeeze(0), 0) 
		dmat+=[b_dmat]
		bmat+=[b_bmat]
		hmat+=[b_hmat]
	dmat, bmat, hmat = torch.stack(dmat, dim=0), torch.stack(bmat, dim=0), torch.stack(hmat, dim=0)

def discriminator_step(b, batch, generator, discriminator, optimizer_d, discriminator_spatial=False, eps=1e-06):
	optimizer_d.zero_grad()
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, \
	op_mask,pedestrians, scene_context, batch_mean, batch_var = batch
	prediction, target, sequence, pedestrians= predict(batch, generator)
	if discriminator_spatial:
		pred_dmat, pred_hmat, pred_bmat = batched_features(prediction)
		target_dmat, target_hmat, target_bmat= batched_features(target)
	
	prediction = prediction-batch_mean.unsqueeze(1).expand_as(prediction)
	prediction = prediction/batch_var.unsqueeze(1).expand_as(prediction)
	target = target-batch_mean.unsqueeze(1).expand_as(target)
	target =target/batch_var.unsqueeze(1).expand_as(target)
	prediction.data.masked_fill_(mask=~op_mask.unsqueeze(-1).expand_as(prediction),value=float(0))
	target.data.masked_fill_(mask=~op_mask.unsqueeze(-1).expand_as(prediction),value=float(0))

	if discriminator_spatial:
		scores_fake = discriminator(prediction, pred_dmat, pred_bmat, pred_hmat, op_mask, domain=generator.spatial_attention.domain) 
	else:
		scores_fake = discriminator(prediction)
	
	fake = Variable(torch.zeros_like(scores_fake), requires_grad=False).uniform_(0.0, 0.3)
	fake_loss = adv_loss(scores_fake, fake)
	
	if discriminator_spatial:
		scores_real = discriminator(target, target_dmat, target_bmat, target_hmat, op_mask, domain=generator.spatial_attention.domain) 
	else:
		scores_real = discriminator(target)
	
	valid = Variable(torch.ones_like(scores_real), requires_grad=False).uniform_(0.7, 1.0)
	
	real_loss = adv_loss(scores_real, valid)
	loss = real_loss+fake_loss
	loss.backward()
	optimizer_d.step()
	
	return loss

def generator_step(b, batch, generator, discriminator=None, optimizer_g=None, best_k=None, weight_sim=None, train=True, discriminator_spatial=False, l2_loss_weight=1):
	if train: 
		optimizer_g.zero_grad()
	min_ade=float(1000)
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, \
	op_mask,pedestrians, scene_context, batch_mean, batch_var = batch
	predictions, target, sequence, pedestrians = predict_multiple(batch, generator, best_k)
	
	for k in range(best_k):
		ade_g, fde_g = eval_metrics(predictions[k], target, pedestrians)
		if ade_g.item()<min_ade:
			min_ade=ade_g
			fde_ = fde_g 
			final_pred = predictions[k]
	if train:
		if not (weight_sim==0):
			predictions = torch.stack(predictions, dim=0)
			similarity_metric = traj_similarity(predictions, op_mask)
		if discriminator_spatial:
			pred_dmat, pred_hmat, pred_bmat = batched_features(final_pred)
				
		final_pred = final_pred - batch_mean.unsqueeze(1).expand(final_pred.size())
		final_pred = final_pred / batch_var.unsqueeze(1).expand(final_pred.size())
		final_pred.data.masked_fill_(mask=~op_mask.unsqueeze(-1).expand(final_pred.size()), value=float(0))
		
		if discriminator_spatial:
			scores_fake = discriminator(final_pred, pred_dmat, pred_bmat, pred_hmat, op_mask, domain=generator.spatial_attention.domain)
		else:
			scores_fake = discriminator(final_pred)
		
		valid = Variable(torch.ones_like(scores_fake), requires_grad=False).uniform_(0.7, 1.0)
		discriminator_loss = adv_loss(scores_fake, valid)
		loss = l2_loss_weight*min_ade+discriminator_loss
		
		if not (weight_sim==0):
			loss = loss+weight_sim*similarity_metric 
		loss.backward()
		optimizer_g.step()
	
	if not train: 
		discriminator_loss=None
	
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
