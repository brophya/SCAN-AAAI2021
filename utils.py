import os
import torch
import math
import numpy as np 
import pandas as pd 
import random
from torch.autograd import Variable
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt 
import torchvision
from torchvision import transforms as transforms 
from PIL import Image 
import seaborn as sns

rad2deg = 180/math.pi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ade(pred, targets, num_peds, eps=1e-14):
    num_peds = num_peds.sum()
    dist=torch.sqrt(((pred-targets)**2).sum(dim=-1)+eps)
    ade=dist.sum()/(num_peds*pred.size(2))
    return ade 

def fde(pred, targets, num_peds, eps=1e-14):
    num_peds = num_peds.sum()
    dist = torch.sqrt(((pred[...,-1,:]-targets[...,-1,:])**2).sum(dim=-1)+eps)
    fde = dist.sum()/(num_peds)
    return fde 

def get_batch(batch):
    batch = [t.to(device) for t in batch]
    if not len(batch[0].size())==4: batch = [t.unsqueeze(0) for t in batch]
    return batch 

def preprocess_image(fname):
    img = Image.open(fname)
    img_size=(224, 224)
    preprocess=transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(img_size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    return img 

def evaluate_model(model, testloader):
    test_ade=float(0)
    test_fde=float(0)
    model.eval()
    for b, batch in enumerate(testloader):
        pred, target,sequence, pedestrians = predict(batch,model)
        ade_b = ade(pred, target, pedestrians)
        fde_b = fde(pred, target, pedestrians)
        test_ade+=ade_b.item()
        test_fde+=fde_b.item()
    test_ade/=(b+1)
    test_fde/=(b+1)
    return test_ade, test_fde 

def predict(batch, net, num_traj=None):
    batch = get_batch(batch)
    sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians, scene_context, batch_mean, batch_var, frame_id = batch 
    target_mask = op_mask.unsqueeze(-1).expand(target.size())
    assert(not(pedestrians.min()==0))
    if num_traj is None:
        pred = net(sequence, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask, scene_context, batch_mean, batch_var)
        batch_mean, batch_var = batch_mean.unsqueeze(1).expand_as(pred), batch_var.unsqueeze(1).expand_as(pred)
        pred = pred*batch_var+batch_mean
        target=target*batch_var+batch_mean 
        target.data.masked_fill_(mask=~target_mask, value=float(0))
        pred.data.masked_fill_(mask=~target_mask, value=float(0))
        return pred,target,sequence[...,:2], pedestrians
    else:
        predictions = []
        for _ in range(num_traj):
            pred =  net(sequence, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask, scene_context, batch_mean, batch_var)
            predictions+=[pred]

        batch_mean, batch_var = batch_mean.unsqueeze(1).expand_as(pred), batch_var.unsqueeze(1).expand_as(pred)
        target=target*batch_var+batch_mean
        target.data.masked_fill_(mask=~target_mask, value=float(0))
        for i in range(len(predictions)):
            predictions[i] = predictions[i]*batch_var+batch_mean
            predictions[i].data.masked_fill_(mask=~target_mask, value=float(0))
        return predictions, target,sequence[...,:2], pedestrians, dist_matrix,bearing_matrix,heading_matrix

def get_distance_matrix(sample, neighbors_dim=0, eps=1e-24):
    if not (neighbors_dim==1): sample=sample.transpose(0,1)
    n=sample.size(1)
    s=sample.size(0)
    norms=torch.sum((sample)**2,dim=2,keepdim=True)
    norms=norms.expand(s,n,n)+norms.expand(s,n,n).transpose(1,2)
    dsquared=norms-2*torch.bmm(sample,sample.transpose(1,2))
    distance = torch.sqrt(torch.abs(dsquared)+eps)
    if not neighbors_dim==1: distance=distance.transpose(0,1)
    return distance 


def get_features(sample, neighbors_dim, previous_sequence=None, mean=None, var=None):
    #print(sample)
    #input("--------")
    if not (mean is None) and not (var is None):
        mean, var = mean.unsqueeze(neighbors_dim).expand_as(sample), var.unsqueeze(neighbors_dim).expand_as(sample)
        sample = sample*var+mean
        if not (previous_sequence is None):
            previous_sequence=previous_sequence*var+mean
    if not (neighbors_dim==1): sample=sample.transpose(0,1)
    n = sample.size(1)
    s = sample.size(0)
    x1 = sample[...,0]
    y1 = sample[...,1]
    x1 = x1.unsqueeze(-1).expand(s, n, n)
    y1 = y1.unsqueeze(-1).expand(s, n, n)
    x2 = x1.transpose(1, 2)
    y2 = y1.transpose(1, 2)
    dx = x2-x1 
    dy = y2-y1 
    bearing=rad2deg*torch.atan2(dy, dx)
    distance=get_distance_matrix(sample,1)
    heading=get_heading(sample, prev_sample=previous_sequence)
    heading = heading.unsqueeze(-1).expand(s, n, n)
    bearing = heading-bearing
    heading = heading.transpose(1,2)-heading
    self_tensor = torch.ones_like(bearing)
    self_tensor[:, range(n), range(n)] = 1
    bearing.data.masked_fill_(mask=~(self_tensor).bool(), value=float(0))
    bearing = torch.where(bearing<0, bearing+360, bearing)
    heading = torch.where(heading<0, heading+360, heading)
    if not neighbors_dim==1: distance, bearing, heading = distance.transpose(0,1),bearing.transpose(0,1),heading.transpose(0,1)
    return distance, bearing, heading

def get_heading(sample, prev_sample=None):
    n=sample.size(1)
    diff=torch.zeros_like(sample)
    if prev_sample is None:
        diff[1:,...]=sample[1:,...]-sample[:-1,...]
        diff[0,...]=diff[1,...]
        heading=rad2deg*torch.atan2(diff[...,1], diff[...,0])
        heading=torch.where(heading<0, heading+360, heading)
    else:
        diff=sample-prev_sample 
        heading = rad2deg * torch.atan2(diff[...,1],diff[...,0])
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

def plot_kde_domain(model, plot_file=None, delta_heading=30, delta_bearing=30):
    r = model.spatial_attention.domain.detach().cpu().numpy()
    max_dist = np.ceil(np.max(r))
    r = np.around(r, 4)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111) #, projection="polar")
    #ax.set_theta_zero_location("N")
     
    for phi in range(np.shape(r)[0]):
        _phi = delta_heading*phi
        r12 = r[phi,:]
        domain = np.zeros(360)
        deg = 0
        for j in range(len(r12)):
            domain[int(deg):int(deg+delta_bearing)] = r12[j]
            deg+=delta_bearing
        offset_bearing=int(delta_bearing/2)
        print(type(offset_bearing))
        domain = np.roll(domain,int(offset_bearing))
        theta = [math.radians(i) for i in np.arange(0, 360, 1)]
        theta = np.array(theta)
        x = domain*np.cos(theta-math.pi/2)
        y = domain*np.sin(theta-math.pi/2)
        x = np.array(x).reshape(360,1)
        y = np.array(y).reshape(360,1)
        xy = np.concatenate((x,y),axis=1)
        df_phi = pd.DataFrame(xy, columns=['x','y'])
        sns.kdeplot(df_phi['x'], df_phi['y'], shade=True, shade_lowest=True)
	#ax.plot(theta, domain, linewidth=1.5, label=str(_phi-offset_bearing)+'$^{o}$-' + str(_phi+delta_heading-offset_bearing)+ '$^{o}$')
    theta_ticks = np.arange(0,360,delta_bearing)
    theta_ticks_labels = [str(int(theta))+"$^{o}$" for theta in theta_ticks]
    #ax.set_xticks([math.radians(theta) for theta in theta_ticks])
    #ax.set_xticklabels(theta_ticks_labels, fontsize=14)
    ax.legend(loc='upper left',bbox_to_anchor=(1.05,1.05),ncol=1,fontsize="large", title="Relative Heading Angle ($\phi^{21}$)")
    ax.tick_params(axis='x', which='major', pad=5)
    ax.tick_params(axis="y", labelsize=14)
    #ax.set_rlabel_position(15)
    ax.annotate('$p_{1}$',xy=(-90,0.5),fontsize=15)
    ax.arrow(0,0,0,1,alpha = 0.5, width = 0.15,edgecolor = 'black', facecolor = 'black', lw = 2)
    ax.grid(linewidth=0.2)
    if plot_file is None:return plt
    else:plt.savefig(plot_file)


