from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import torch
import os
import warnings
from torch.autograd import Variable
import math
import numpy as np
import pandas as pd
import datetime
import random
import matplotlib
matplotlib.use("agg")
import matplotlib.patches as mpatches
from matplotlib import cm
from termcolor import colored
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import glob

np.set_printoptions(precision=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rad2deg = 180/math.pi

argoverse=False

# old seed 100 OR 120
# for 32--> 100
# for 4 --> 1
def seed_everything(seed=100):
	"""
	Seeds everything 
	"""
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.initial_seed()
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic=True


def get_batch(batch):
	"""
	Batch processing: all tensors to cuda if cuda is available and converts to batch first for batch size=1
	Returns:
		batch : Next batch 
	"""
	batch = [tensor.to(device) for tensor in batch]
	if not len(batch[0].size())==4: # not batch-first
		batch = [tensor.unsqueeze(0) for tensor in batch]
	return batch

def predict(batch,net):
	"""
	Predicts intent for one batch
	Args:
		Batch: Input Batch 
		Net: Model for prediction
	Returns:
		pred (Variable): Model prediction for given batch (batch_size, num_pedestrians, prediction_length, output_size)
		target (Variable): ground truth for given batch (batch_size, num_pedestrians, prediction_length, output_size)
		sequence (Variable): input sequence (batch_size, num_pedestrians, sequence_length, feature_size)
		temporal_attention_dict (Dict): Alignment Vectors for given batch
	""" 
	batch = get_batch(batch)
	if len(batch)==8:
		sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians = batch 
		agent_idx = torch.zeros_like(pedestrians)
	else:
		sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians, agent_idx = batch		
	#assert(not(pedestrians==0).any()), "No pedestrians in frame"
	sequence, dist_matrix, bearing_matrix, heading_matrix = Variable(sequence, requires_grad=True), Variable(dist_matrix, requires_grad=True), Variable(bearing_matrix, requires_grad=True), Variable(heading_matrix, requires_grad=True)
	pred = net(sequence, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask)
	#assert(not torch.isnan(pred).any())
	target_mask = op_mask.unsqueeze(-1).expand(target.size())
	target.data.masked_fill_(mask=~target_mask, value=float(0))
	pred.data.masked_fill_(mask=~target_mask, value=float(0))
	#target = torch.where(~target_mask, torch.zeros_like(target), target)
	#pred = torch.where(~target_mask, torch.zeros_like(pred), pred)
	return pred,target,sequence[...,:2], pedestrians, agent_idx
	
def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	os.system("rm -r tmp")
	return np.argmax(memory_available).item()

#def get_distance_matrix(sample,neighbors_dim=0,eps=0):
def get_distance_matrix(sample,neighbors_dim=0,eps=1e-14):
	"""
	Distance matrix computation
	Args:
		sample (Variable): one batch or one sample of input data (batch_size, num_pedestrians, feature_size) OR (num_pedestrians, sequence_length, feature_size)
		neighbors_dim (int): dimension for num_pedestrians in sample to distinguish b/w batch and sample
		eps (float): very small eps value to avoid nan gradients from torch.sqrt in distance computation
	Returns:
		distance (Variable): distance matrix (batch_size, num_pedestrians, num_pedestrians) in meters if neighbors_dim=1 else (num_pedestrians, sequence_length, num_pedestrians) if neighbors_dim=0
	"""
	#print(sample.size())
	if not neighbors_dim==1:
		sample=sample.transpose(0,1)
	n = sample.size(1)
	s = sample.size(0)
	norms=torch.sum((sample)**2,dim=2,keepdim=True)
	norms=norms.expand(s,n,n)+norms.expand(s,n,n).transpose(1,2)
	dsquared=norms-2*torch.bmm(sample,sample.transpose(1,2))
	distance = torch.sqrt(torch.abs(dsquared)+eps)
	if argoverse is True: 
		distance=3000*distance 
	else:
		distance = 15*distance 
	if not neighbors_dim==1:
		distance=distance.transpose(0,1)
	return distance

def get_features(sample,neighbors_dim,previous_sequence=None):
	"""
	Feature matrices computation
	Args:
		sample (Variable): one batch or one sample of input data (batch_size, num_pedestrians, feature_size) OR (num_pedestrians, sequence_length,     feature_size)
		delta_rb (float): relative bearing discretization angle in degrees
		delta_heading (float): relative heading discretization angle in degrees
		neighbors_dim (int): dimension for num_pedestrians in sample to distinguish b/w batch and sample
		previous_sequence (Variable): previous prediction to compute heading angle 
	Returns:
		distance (Variable): distance matrix (batch_size, num_pedestrians, num_pedestrians) or (num_pedestrians, sequence_length, num_pedestrians)
		bearing (Variable): relative bearing matrix (batch_size, num_pedestrians, num_pedestrians) or (num_pedestrians, sequence_length, num_pedestrians)
		heading (Variable): relative heading matrix (batch_size, num_pedestrians, num_pedestrians) or (num_pedestrians, sequence_length, num_pedestrians)
	"""
	
	offset_bearing, offset_heading = 0,0
	if not (offset_heading==0 and offset_bearing==0):
		warnings.warn("all computation for features with offsets: " + str(offset_bearing) + " " + str(offset_heading))
	if not neighbors_dim==1:	
		sample=sample.transpose(0,1)
	n = sample.size(1)
	s = sample.size(0)
	
	x1 = sample[...,0]
	
	y1 = sample[...,1]
	
	x1 = x1.unsqueeze(-1).expand(s,n,n)
	y1 = y1.unsqueeze(-1).expand(s,n,n)
	x2 = x1.transpose(1,2)
	y2 = y1.transpose(1,2)
	dx = x2-x1
	
	dy = y2-y1
	
	bearing = rad2deg * (torch.atan2(dy,dx))
	
	bearing = torch.where(bearing<0, bearing+360, bearing)
	
	distance = get_distance_matrix(sample,1)
	
	heading = get_heading(sample,prev_sample=previous_sequence)
	
	heading = heading.repeat(1,n).view(s, n, n)
	#heading = heading.unsqueeze(-1).expand(s,n,n)
	bearing=bearing-heading
	
	assert(not(torch.isnan(distance)).any()),"distance matrix has nans!"
	bearing = torch.where(distance.data==distance.data.min(), torch.zeros_like(bearing), bearing)
	
	heading = heading.transpose(1,2)-heading
	bearing = torch.where(bearing<0, bearing+360, bearing)
	heading = torch.where(heading<0, heading+360, heading)
	
	if not neighbors_dim==1:
		distance, bearing, heading = distance.transpose(0,1),bearing.transpose(0,1),heading.transpose(0,1)
	
	return distance, bearing, heading

def get_heading(sample,prev_sample=None):
	"""
	Heading matrix computation
	Args:
		sample (Variable): one batch or one sample of input data (batch_size, num_pedestrians, feature_size) OR (sequence_length, num_pedestrians, feature_size)
		prev_sample (Variable): previous prediction to compute heading angle
	Returns:
		heading (Variable): absolute heading matrix (batch_size, num_pedestrians) or (sequence_length, num_pedestrians)
	"""
	n = sample.size(1)
	diff = torch.zeros_like(sample)
	if prev_sample is None:
		diff[1:,...] = sample[1:,...]-sample[:-1,...]
		diff[0,...] = diff[1,...] 
		#diff[0,...].data.copy_(diff[1,...])
		heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
		heading = torch.where(heading<0, heading+360, heading)
		#if (heading==0).any():
		#	heading = fill_zeros(heading)
	else:
		diff = sample-prev_sample
		if torch.isnan(sample).any(): print("Sample has nans")
		if torch.isnan(prev_sample).any(): print("prev has nans")
		
		heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
		
		heading = torch.where(heading<0, heading+360, heading)
		
	return heading

def fill_zeros(heading):
	"""
	Interpolate zeros in heading matrix
	Args:
		heading (Variable): absolute heading matrix (batch_size, num_pedestrians) or (sequence_length, num_pedestrians)
	Returns:
		heading (Variable): Interpolated heading matrix (batch_size, num_pedestrians) or (sequence_length, num_pedestrians)
	"""
	heading_np = heading.detach().cpu().numpy()
	heading_pd = pd.DataFrame(heading_np)
	heading_pd = heading_pd.replace(to_replace=0, method="ffill").replace(to_replace=0, method="bfill")
	return torch.from_numpy(heading_pd.values).to(heading) 
	
#	print(heading_pd)
#	input("pause..")
	"""
	neighbors = heading_np.shape[1]
	slen = heading_np.shape[0]
	for n in range(neighbors):
		if not (heading_np[:,n]==0).any():
			continue
		idx = np.arange(slen)
		idx[heading_np[:,n]==0]=0
		idx = np.maximum.accumulate(idx,axis=0)
		print(idx)
		heading_np[:,n] = heading_np[idx,n]
		print(heading_np) 
		if (heading_np[:,n]==0).any():
			idx = np.arange(slen)
			idx[heading_np[:,n]==0]=0
			idx = np.minimum.accumulate(idx[::-1],axis=0)
			print(idx)
			heading_np[:,n] = heading_np[idx[::-1],n]
	"""
#	return torch.from_numpy(heading_np).to(heading)

def get_color_map(x,y):
	shifted_x = [0]
	shifted_y = [0]
	for i in range(len(x)):
		shifted_x.append(x[i])
		shifted_y.append(y[i])
	speed = (x-shifted_x[:-1])**2+(y-shifted_y[:-1])**2
	speed = (speed-np.min(speed))/(np.max(speed)-np.min(speed))
	print(np.shape(speed))
	return speed
	
def convert_tensor_to_numpy(tensor_):
	"""
	Conversion of tensor to numpy array
	Args:
		tensor_ (Variable)
	Returns:
		Numpy Array containing tensor_ data detached from gradient 
	"""
	return np.array(tensor_.clone().detach().cpu().numpy())


class Plotter(object):
	def __init__(self,args):
		"""
		Plotter to log training and validation error progress during training
		"""	
		super(Plotter,self).__init__()
		plot_dir='plots/'
		self.filename = "{}_enc{}_dec{}_mlp{}_att{}_domain{}_emb{}_difflr{}_dropframes{}_{}dropout_{}".format(args.model,args.enc_dim, args.dec_dim, args.mlp_dim, args.att_dim, args.param_domain, args.embedding_dim, args.diff_lr, args.drop_frames, args.dropout, args.init_type)
		self.train_plots = "{}/{}/{}.png".format(plot_dir, args.data, self.filename)
		self.train_arr_fname = "train_arrays/{}/{}.txt".format(args.data, self.filename) 
		if not os.path.isdir(plot_dir+str(args.data)):
			os.makedirs(plot_dir+str(args.data))
		if not os.path.isdir("train_arrays/{}".format(args.data)):
			os.makedirs("train_arrays/{}".format(args.data))
		print(colored("Plotting training and validation error in %s" %(self.train_plots),"blue"))
		self.train=[]
		self.valid=[]
		self.test=[]
	def update(self,train_loss,valid_loss,test_loss,best_test):
		"""
		Update train and validation error for current epoch
		Args:
			train_loss (float): train error for current epoch
			valid_loss (float): validation error for current epoch
		Returns:
			updated error plot saved to plot_file
		"""
		self.train.append(train_loss)
		self.valid.append(valid_loss)
		self.test.append(test_loss)
		fig = plt.figure()
		plt.plot(self.train, linewidth=2, label='Train ADE')
		plt.plot(self.valid, linewidth=2, label='Valid ADE')
		plt.plot(self.test, linewidth=2, label='Test ADE')
		plt.legend()
		plt.title("Test ADE: {:.3f}".format(best_test))
		plt.xlabel("Epoch")
		plt.savefig(self.train_plots)
		np.savetxt(self.train_arr_fname, self.train, delimiter="\n") 
		plt.close()

def print_lr(optimizer):
	"""
	Print learning rates after every few epochs
	Args:
		optimizer : optimizer used for training
	"""
	print(colored("******* Current Learning Rates ********\n{}".format([param['lr'] for opt in optimizer for param in opt.param_groups]),"green"))

def inspect_gradient_flow(model):
	"""
	Visualize backward flow of gradients during loss.backward() for anomaly detection
	Args:
		model 
	Returns:
		plot of gradients
	"""
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in list(model.named_parameters()):
		if(p.requires_grad) and (not p.grad is None) and ("bias" not in n):
		#	print(n)
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
			max_grads.append(p.grad.abs().max())
	fig = plt.figure(figsize=(12,12))
	plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
		Line2D([0], [0], color="b", lw=4),
		Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	plt.savefig("gradient.png")
	plt.close()

def check_gradients(model):
	"""
	Gradient Anomaly Detection (NaN or None)
	"""
	for name, param in model.named_parameters():		
		if torch.isnan(param.grad).any():
			print(colored("{} gradient is NaN!".format(name),"red"))
		elif param.grad is None:
			print(colored("{} gradient is None!".format(name),"red"))

def check_nan(tensor):
	"""
	Throws Assertion Error if NaN in tensor 
	Args:
		tensor
	"""
	assert(not(torch.isnan(tensor).any()))

def custom_init(param, param_domain):
	val = np.linspace(0.1, param_domain, 6)
	print(val)
	for col in range(param.size(1)):
		if (col==0) or (col==11):
			param[:,col].data.fill_(val[5])
		elif (col==1) or (col==10):
			param[:,col].data.fill_(val[4])
		elif (col==2) or (col==9):
			param[:,col].data.fill_(val[3])
		elif (col==3) or (col==8):
			param[:,col].data.fill_(val[2])
		elif (col==4) or (col==7):
			param[:,col].data.fill_(val[1])
		else:
			param[:,col].data.fill_(val[0])

def plot_domain(model, plot_file, delta_heading, delta_bearing):
	r = model.spatial_attn.domain.detach().cpu().numpy()
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
		theta = [math.radians(i) for i in np.arange(0, 360, 1)]
		ax.plot(theta, domain, linewidth=1.5, label=str(_phi)+'$^{o}$-' + str(_phi+delta_heading)+ '$^{o}$')
	theta_ticks = np.arange(0,360,delta_bearing)
	theta_ticks_labels = [str(int(theta))+"$^{o}$" for theta in theta_ticks]
	ax.set_xticks([math.radians(theta) for theta in theta_ticks])
	ax.set_xticklabels(theta_ticks_labels, fontsize=14) 
	
	ax.legend(loc='upper left',bbox_to_anchor=(1.05,1.05),ncol=1,fontsize="large", title="Relative Heading Angle ($\phi^{21}$)")
	ax.set_rmax(8)
	ax.tick_params(axis='x', which='major', pad=5)
	ax.tick_params(axis="y", labelsize=14)
	ax.set_rlabel_position(130)
	ax.annotate('$p_{1}$',xy=(-90,0.5),fontsize=15)
	ax.arrow(0,0,0,1,alpha = 0.5, width = 0.15,edgecolor = 'black', facecolor = 'black', lw = 2)
	ax.grid(linewidth=0.2)	
	
	plt.savefig(plot_file)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
