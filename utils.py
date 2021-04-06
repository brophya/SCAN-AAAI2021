import os
import torch
import math
import csv
import numpy as np 
import pandas as pd 
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torchvision
from torchvision import transforms as transforms 
from PIL import Image 
import seaborn as sns

from tqdm import tqdm

rad2deg = 180/math.pi
deg2rad = math.pi/180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps=1e-14

def round_(tensor, digits=2):
	rounded=(tensor*10**digits).round()/(10**digits)
	return rounded

def eval_metrics(pred, targets, num_peds, mask, eps=1e-14, reduction='mean'):
    num_peds = num_peds.sum()
    dist=torch.sqrt(((pred-targets)**2).sum(dim=-1)+eps) 
    dist = dist*mask
    dist_final = dist[:,:,-1]
    fde = dist_final.sum()
    ade = dist.sum()
    if reduction=='mean': 
        ade = ade.div(num_peds*pred.size(2)) 
        fde = fde.div(num_peds)
    return ade , fde

def fde(pred, targets, num_peds, eps=1e-14):
    num_peds = num_peds.sum()
    dist = torch.sqrt(((pred[...,-1,:]-targets[...,-1,:])**2).sum(dim=-1)+eps)
    fde = dist.sum()/(num_peds)
    return fde 

def get_batch(batch):
    """
    Function to move tensors to gpu if available else cpu 
    and add batch dimension if it doesn't exist
    """
    batch = [t.to(device) for t in batch]
    if not len(batch[0].size())==4: batch = [t.unsqueeze(0) for t in batch]
    return batch 

def preprocess_image(fname):
    """
    Function to preprocess scene 
    """
    img = Image.open(fname)
    img_size=(224, 224)
    preprocess=transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(img_size), 
        transforms.functional.to_grayscale,
	transforms.ToTensor()])

        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])
    img = preprocess(img)
    return img 

def evaluate_model(model, testloader):
    """
    Function to evaluate trained model 'model' given 'testloader'
    """
    test_ade=float(0)
    test_fde=float(0)
    model.eval()
    total_pedestrians=float(0) 
    for b, batch in enumerate(testloader):
        pred, target,sequence, pedestrians, op_mask, _ = predict(batch,model)
        total_pedestrians+=pedestrians.sum() 
        ade_b, fde_b = eval_metrics(pred, target, pedestrians, op_mask, eps=0, reduction='sum')
        test_ade+=ade_b.item()
        test_fde+=fde_b.item()
    test_ade/=(total_pedestrians*pred.size(2))
    test_fde/=(total_pedestrians)
    return test_ade, test_fde

def predict(batch, net, domain=None):
    """
    Function to predict joint future trajectories for all samples in the batch
    using model 'net'
    """
    batch = get_batch(batch)
    sequence,target,dist_matrix,bearing_matrix,heading_matrix,\
    ip_mask,op_mask,pedestrians, scene_context, mean, var = batch
    target_mask = op_mask.unsqueeze(-1).expand(target.size())
    pred = net(sequence, pedestrians, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask, scene_context, mean, var,domain=domain)
    pred = revert_orig_tensor(pred, mean, var, op_mask, dim=1)
    target = revert_orig_tensor(target, mean, var, op_mask, dim=1)
    sequence = revert_orig_tensor(sequence, mean, var, ip_mask, dim=1)
    return pred, target, sequence, pedestrians, op_mask, ip_mask 

def normalize_tensor(tensor, mean, var, mask, dim=0):
	"""
	Normalizes tensor to 0,1
	"""
	var_ = var.unsqueeze(dim).expand_as(tensor) 
	tensor=tensor/var_
	mask_ = mask.unsqueeze(-1).expand_as(tensor) 
	tensor = tensor*mask_
	return tensor

def revert_orig_tensor(tensor, mean, var, mask, dim=0):
	"""
	Reverts normalized tensor to original value
	"""
	mean_ = mean.unsqueeze(dim).expand_as(tensor)
	var_ = var.unsqueeze(dim).expand_as(tensor)
	tensor=tensor*var_
	mask_ = mask.unsqueeze(-1).expand_as(tensor)
	tensor = tensor*mask_
	return tensor

def get_distance_matrix(sample, neighbors_dim=0, mask=None, eps=1e-14):
    """
    sample -> batch_size x num_pedestrians x 2 OR sequence_length x num_pedestrians x 2 
    neighbors_dim -> dimension in sample pertaining to num_pedestrians 
    
    computes distance matrix -> batch_size x num_pedestrians x num_pedestrians OR sequence_length x num_pedestrians x num_pedestrians
    """
    s, n = sample.size()[:2]
    norms=torch.sum((sample)**2,dim=-1,keepdim=True)
    norms=norms.expand(s, n, n)+norms.expand(s, n, n).transpose(1,2)
    ab_term = torch.bmm(sample,sample.transpose(1,2))
    dsquared=norms-2*ab_term.view(s,n,n)
    distance = torch.sqrt(torch.abs(dsquared)+eps)
    return distance 

def mask_matrix(matrix, mask, n_dims):
	"""
	mask matrix given mask
	"""
	mask = mask.unsqueeze(-1).expand_as(matrix)
	mask = mask.mul(mask.transpose(*n_dims))
	matrix = matrix*mask
	return matrix

def get_features(sample, neighbors_dim, previous_sequence=None, mean=None, var=None, mask=None, eps=1e-14):
    """
    Returns distance matrix, relative bearing matrix, relative heading matrix
    Given sample -> sequence_length x num_pedestrians x 2 
    OR
    Given sample and previous_sequence -> batch_size x num_pedestrians x 2

    Given p1 at (x1, y1) and p2 at (x2, y2) at t, 
    absolute bearing is computed as atan2(y2-y1/x2-x1) 

    Given p1 at (x1, y1) at t-1 and p1 at (x2, y2) at t
    absolute heading is computed as atan2(y2-y1/x2-x1)

    relative bearing is computed as the difference between absolute bearing and heading of p1

    relative heading is computed as the difference between heading of p2 and heading of p1 
    """
    if not (mean is None) and not (var is None):
        mean, var = mean.unsqueeze(neighbors_dim).expand_as(sample), var.unsqueeze(neighbors_dim).expand_as(sample)
        sample = (sample)*var+mean
        if not (previous_sequence is None):
            previous_sequence=(previous_sequence)*var+mean
    if not (neighbors_dim==1): sample=sample.transpose(0,1)
    s, n = sample.size()[:2]
    if len(sample.size())==4: # batch_size x num_pedestrians x pred_len x 2 
        plen=sample.size(2)
        expand_dims = (s, n, plen, n)
        n_dims = (1, 3)
    elif len(sample.size())==3:
        expand_dims=(s, n, n)
        n_dims = (1,2)
    x1 = sample[...,0] # x for all pedestrians
    y1 = sample[...,1] # y for all pedestrians 
    x1 = x1.unsqueeze(-1).expand(*expand_dims)
    y1 = y1.unsqueeze(-1).expand(*expand_dims)
    x2 = x1.transpose(*n_dims)
    y2 = y1.transpose(*n_dims)
    dx = x2-x1 # x for all pedestrians diff. 
    dy = y2-y1 # y for all pedestrians diff. 
    bearing=torch.atan2(dy, dx) # absolute bearing 
    bearing=rad2deg*bearing
    bearing = torch.where(bearing<0, bearing+360, bearing)
    if len(sample.size())==4:
        distance=torch.stack([get_distance_matrix(sample[i,...].transpose(0,1),neighbors_dim=1, mask=mask, eps=eps) for i in range(sample.size(0))], dim=0)
        distance=distance.transpose(1,2)
    else:
        distance=get_distance_matrix(sample,neighbors_dim=1, mask=mask, eps=eps)
    heading=get_heading(sample, prev_sample=previous_sequence, mask=mask) 
    heading = heading.unsqueeze(-1).expand(*expand_dims) 
    heading = torch.where(heading<0, heading+360, heading)
    bearing=bearing-heading # 
    heading = heading.transpose(*n_dims)-heading
    self_tensor = torch.zeros_like(bearing)
    self_tensor[:, range(n), ..., range(n)] = 1
    bearing.data.masked_fill_(mask=(self_tensor).bool(), value=float(0))
    bearing = torch.where(bearing<0, bearing+360, bearing)
    heading = torch.where(heading<0, heading+360, heading)
    if not mask is None:
        bearing, heading, distance = mask_matrix(bearing, mask, n_dims), mask_matrix(heading, mask, n_dims), mask_matrix(distance, mask, n_dims)
    if not neighbors_dim==1: distance, bearing, heading = distance.transpose(0,1),bearing.transpose(0,1),heading.transpose(0,1)
    return distance, bearing, heading

def get_heading(sample, prev_sample=None, mask=None):
    """
    Computes heading matrix
    """
    n=sample.size(1)
    diff=torch.zeros_like(sample)
    if prev_sample is None:
        if (len(sample.size())==3): 
            diff[1:,...]=sample[1:,...]-sample[:-1,...]
            diff[0,...] = 2*math.pi*torch.rand(diff[0,...].size()).to(sample.device)+math.pi
            diff[0,...]=diff[1,...]
        elif (len(sample.size())==4):
            diff[:,:,1:,...] = sample[:,:,1:,...]-sample[:,:,:-1,...]
            diff[:,:,0,...] = diff[:,:,1,...]
    else:
        diff=sample-prev_sample # y - y', x - x' (own displacement)
    heading = rad2deg * torch.atan2(diff[...,1],diff[...,0]) #+(1e-14)) # absolute heading
    return heading 

def plot_domain(model, plot_file, delta_heading, delta_bearing):
    r = model.spatial_attention.domain.detach().cpu().numpy()
    max_dist = np.ceil(np.max(r))
    r = np.around(r, 4)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    for phi in range(np.shape(r)[0]):
        _phi = delta_heading*phi
        r12 = r[phi,:]
        domain = np.zeros(360)
        deg = 0
        for j in range(len(r12)):
            domain[int(deg):int(deg+delta_bearing)] = r12[j]
            deg+=delta_bearing
        offset_bearing=int(delta_bearing/2)
        domain = np.roll(domain,offset_bearing)
        theta = [math.radians(i) for i in np.arange(0, 360, 1)]
        ax.plot(theta, domain, linewidth=1.5, label=str(_phi)+'$^{o}$-' + str(_phi+delta_heading)+ '$^{o}$')
    theta_ticks = np.arange(0,360,int(delta_bearing/2))
    theta_ticks_labels = [str(int(theta))+"$^{o}$" for theta in theta_ticks]
    ax.set_xticks([math.radians(theta) for theta in theta_ticks])
    ax.set_xticklabels(theta_ticks_labels, fontsize=14)
    ax.legend(loc='upper left',bbox_to_anchor=(1.05,1.05),ncol=1,fontsize="large", title="Relative Heading Angle ($\phi^{21}$)")
    ax.tick_params(axis='x', which='major', pad=5)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_rlabel_position(15)
    ax.annotate('$p_{1}$',xy=(-90,0.5),fontsize=15)
    ax.arrow(0,0,0,1,alpha = 0.5, width = 0.15,edgecolor = 'black', facecolor = 'black', lw = 2)
    ax.grid(linewidth=0.2)
    plt.savefig(plot_file)

def evaluate_collisions(testdataset,net,netfile,test_batch_size,thresholds):
	net.eval()
	ade = float(0)
	mean_error = float(0)
	fde = float(0)
	testloader = DataLoader(testdataset,batch_size=test_batch_size,collate_fn=collate_function(),shuffle=True)
	numTest=len(testloader)
	coll_array=[]
	for threshold in thresholds:
		print("Evaluating collisions for threshold: {}".format(threshold))
		with torch.no_grad():
			num_coll = float(0)
			num_total = float(0)
			for b, batch in enumerate(testloader):
				pred, target, sequence, context_vector, pedestrians = predict(batch,net)
				pred = pred.squeeze(0).permute(1,0,2)
				dist_matrix = get_distance_matrix(pred,neighbors_dim=1) 
				count = torch.where(dist_matrix<threshold, torch.ones_like(dist_matrix), torch.zeros_like(dist_matrix))
				count = count.sum()-pedestrians*pred.size(0)
				count = count/2 # each collision is counted twice
				count = count.item()
				if (count>0):
					num_coll+=1
				num_total += 1 
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}")
			num_coll_percent = (num_coll/num_total)*100
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}; % collisions: {num_coll_percent}%") 
		coll_array+=[num_coll_percent]
	return coll_array


def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	# os.system("rm -r tmp")
	return np.argmax(memory_available) 

def init_weights(m):
    """
    initializes all weights in all linear layers to xavier normal initialization 
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1: nn.init.xavier_normal_(m.weight)

class EarlyStopping:
	def __init__(self, patience=40, delta=0.0001):
		self.patience=patience
		self.counter=0
		self.val_loss_min=np.Inf
		self.delta=delta
		self.early_stop=False
		self.best_score=None
	def __call__(self, val_loss):
		score=-val_loss
		if not self.best_score is None and score<(self.best_score+self.delta):
			self.counter+=1
			if self.counter>=self.patience:
				self.early_stop=True
		else:
			self.best_score=score
			self.counter=0



