from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import argparse

import numpy as np
from torch.utils.data import Dataset, DataLoader

from arguments import *
from model import *
from data import *
from metrics import *
from losses import *
from generative_utils import *

import matplotlib
matplotlib.use("agg")
import imageio
import seaborn as sns

torch.set_printoptions(precision=2)        

from matplotlib import pyplot as plt 

args = parse_arguments()

def get_df(traj_p):
	df = []
	columns = ['ix', 't', 'x', 'y']
	for ix in range(traj_p.size(0)):
		for t in range(traj_p.size(1)):
			df.append([ix, t, traj_p[ix, t, 0].item(), traj_p[ix, t, 1].item()])

	df=pd.DataFrame(df, columns=columns)
	return df


seed=100
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.initial_seed()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.gpu_id:
	torch.cuda.set_device(args.gpu_id)      

generator = TrajectoryGenerator(args.model_type, args.obs_len, args.pred_len, args.ip_dim, args.op_dim, args.embedding_dim, args.encoder_dim, args.decoder_dim, args.attention_dim, device, args.domain_type, args.param_domain, args.delta_bearing, args.delta_heading, args.domain_init_type, noise_dim=args.noise_dim, noise_type=args.noise_type)
discriminator = TrajectoryDiscriminator(args.obs_len, args.pred_len, args.embedding_dim, args.encoder_dim, args.delta_bearing, args.delta_heading, args.attention_dim, device=device, domain_type=args.domain_type, param_domain=args.param_domain, domain_init_type=args.domain_init_type)
generator = generator.float().to(device)
discriminator = discriminator.float().to(device)
print(len(glob.glob("data/{}/test/*.txt".format(args.dset_name))))
testdataset = dataset(glob.glob("data/{}/test/*.txt".format(args.dset_name)), args)
testloader = DataLoader(testdataset, batch_size = 1, collate_fn = collate_function(), shuffle=True)
g_file = "./trained-models/{}/{}/g_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim ,args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
d_file = "./trained-models/{}/{}/d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim, args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
generator.load_state_dict(torch.load(g_file+".pt"))
discriminator.load_state_dict(torch.load(d_file+".pt"))


generator.eval()
ade = float(0)
fde = float(0)

"""
if not os.path.exists("plots/{}/{}-{}".format(args.dset_name, args.weight_sim, args.best_k)): 
	os.makedirs("plots/{}/{}-{}".format(args.dset_name, args.weight_sim, args.best_k))
"""
pairwise=True
with torch.no_grad():
	for b, batch in enumerate(testloader):
		if not (b==4):
			continue
		if (b>4):
			exit()
		batch = [tensor.to(device) for tensor in batch] 
		sequence, target, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask, ped_count = batch
		if torch.sum(ped_count)>5 and pairwise is False:
			continue 
		sequence, target, distance_matrix, bearing_matrix, heading_matrix = sequence.float(),target.float(), distance_matrix.float(), bearing_matrix.float(), heading_matrix.float()
		ip_mask, op_mask = ip_mask.bool(), op_mask.bool()     
		traj = []
		for i in range(args.num_traj):
			out_g = generator(sequence, distance_matrix, bearing_matrix, heading_matrix, ip_mask, op_mask)   
			input_d = torch.cat([sequence, out_g], dim=2)
			target_d, target_b, target_h = get_features(out_g, 1) 
			dmat, bmat, hmat = torch.cat([distance_matrix, target_d], dim=2), torch.cat([bearing_matrix, target_b], dim=2), torch.cat([heading_matrix, target_h], dim=2)  
			scores = discriminator(input_d, dmat, bmat, hmat, ip_mask, op_mask)  
			#print("Trajectory: {}".format(i))
			#print(scores.min(), scores.max())
			#print(f"Trajectory: {i} : Ped1: {scores[1].item()}, Ped2: {scores[4].item()}")
			op_mask_ = op_mask.unsqueeze(-1).expand_as(out_g)  
			out_g.data.masked_fill_(mask=~op_mask_, value=float(0))  
			target.data.masked_fill_(mask=~op_mask_, value=float(0)) 
			traj+=[out_g]
		
		if args.plot_densities:
			print(f"Plotting density plot for sample {b}")
			seq_b = sequence.squeeze(0).clone().detach().cpu()
			target_b = target.squeeze(0).clone().detach().cpu() 
			traj = torch.stack(traj, dim=0).clone().detach().cpu().squeeze(1)
			num_ped, slen = seq_b.size()[:2] 
			if pairwise is False: 
				fig = plt.figure()   
				colors = ['red', 'blue', 'orange', 'green', 'yellow']
				#colors = plt.cm.rainbow(np.linspace(0, 1, num_ped))
			for p1 in range(num_ped):  
				seq_p1 = seq_b[p1,...].squeeze(0)      
				gt_p1 = target_b[p1,...].squeeze(0)  
				traj_p1 = traj[:,p1,...].squeeze(0) 
				df1 = get_df(traj_p1)       
				mean_x1 = traj_p1[...,0].mean(dim=0)  
				mean_y1 = traj_p1[...,1].mean(dim=0) 
				mean_x1 = torch.cat((seq_p1[...,-1,0].unsqueeze(0), mean_x1), dim=0)
				mean_y1 = torch.cat((seq_p1[...,-1,1].unsqueeze(0), mean_y1), dim=0)
				gt_p1 = torch.cat((seq_p1[...,-1,:].unsqueeze(0), gt_p1), dim=0)
				if pairwise is True:
					if not (p1==1):
						continue
					for p2 in range(p1+1, num_ped):
						if not (p2==3):
							continue
						fig = plt.figure()   
						colors = plt.cm.rainbow(np.linspace(0, 1, 2))
						seq_p2 = seq_b[p2,...].squeeze(0) 
						gt_p2 = target_b[p2,...].squeeze(0) 
						traj_p2 = traj[:,p2,...].squeeze(0)
						df2 = get_df(traj_p2)  
						mean_x2 = traj_p2[...,0].mean(dim=0)    
						mean_y2 = traj_p2[...,1].mean(dim=0) 
						mean_x2 = torch.cat((seq_p2[...,-1,0].unsqueeze(0), mean_x2), dim=0)
						mean_y2 = torch.cat((seq_p2[...,-1,1].unsqueeze(0), mean_y2), dim=0) 
						gt_p2 = torch.cat((seq_p2[...,-1,:].unsqueeze(0),gt_p2),dim=0)    
						sns.lineplot(15*seq_p1[...,0], 15*seq_p1[...,1],marker="o",markersize=4, color=colors[0])    
						sns.lineplot(15*gt_p1[...,0], 15*gt_p1[...,1], style=True, dashes=[(2,2)], color=colors[0])
						sns.lineplot(15*mean_x1, 15*mean_y1, marker="o", markersize=4, color=colors[0])
						sns.kdeplot(15*df1['x'], 15*df1['y'], color=colors[0], shade=True, shade_lowest=False) 
						sns.lineplot(15*seq_p2[...,0], 15*seq_p2[...,1],marker="o",markersize=4, color=colors[1])
						sns.lineplot(15*gt_p2[...,0], 15*gt_p2[...,1], style=True, dashes=[(2,2)], color=colors[1])
						sns.lineplot(15*mean_x2, 15*mean_y2, marker="o", markersize=4, color=colors[1])
						sns.kdeplot(15*df2['x'], 15*df2['y'], color=colors[1], shade=True, shade_lowest=False)
						plt.xlim([-6, 9])
						plt.ylim([-2, 1])
						plt.xlabel('x', fontsize=16)
						plt.ylabel('y', fontsize=16)
						plt.tick_params(labelsize=16)
						plt.legend([],[], frameon=False)
						plt.title("{}V-{}".format(args.best_k, args.weight_sim), fontsize=20)
						plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.01)
						plt.tight_layout()
						plt.savefig("plots/{}_{}_{}.pdf".format(args.dset_name, args.weight_sim, args.best_k))
						plt.close()
				else:
					sns.lineplot(15*seq_p1[...,0], 15*seq_p1[...,1],marker="o",color=colors[p1]) 
					sns.lineplot(15*mean_x1, 15*mean_y1, marker="o", color=colors[p1])
					sns.kdeplot(15*df1['x'], 15*df1['y'], color=colors[p1], shade=True, shade_lowest=False) 
			if pairwise is False:
				plt.savefig("plots/{}/{}/{}/{}.png".format(args.dset_name, args.weight_sim, args.best_k, b))
				plt.close()
		if args.plot_gifs:
			pairwise=False
			seq_b = sequence.squeeze(0).clone().detach().cpu()
			target_b = target.squeeze(0).clone().detach().cpu()
			traj = torch.stack(traj, dim=0).clone().detach().cpu().squeeze(1)
			num_ped, slen = seq_b.size()[:2]
			disp = ((traj[...,0,-1]-traj[...,0,0])**2+(traj[...,1,-1]-traj[...,1,0])**2).sum(dim=1)
			traj_ix = [disp.argmax(), disp.argmin()]

			#exit()
			if (num_ped>5):
				continue
			if pairwise is False:
				
				colors = ['red', 'blue', 'orange', 'green', 'cyan']
				for t in traj_ix:
					fig, ax  = plt.subplots()
					for p1 in range(num_ped):
						seq_p1 = seq_b[p1,...].squeeze(0) 
						gt_p1 = target_b[p1,...].squeeze(0)  
						traj_p1 = traj[:,p1,...].squeeze(0) 
						#print((traj_p1[...,0,-1]-traj_p1[...,0,0])**2 + (traj_p1[...,1,-1]-traj_p1[...,1,0])**2)
						gt_p1 = torch.cat((seq_p1[...,-1,:].unsqueeze(0),gt_p1),dim=0)  
						pred1 = traj_p1[t,...]
						pred1 = torch.cat((seq_p1[...,-1,:].unsqueeze(0), pred1), dim=0)
						ax.plot(15*seq_p1[...,0], 15*seq_p1[...,1], marker="o", markersize=4,color=colors[p1])
						ax.plot(15*pred1[...,0], 15*pred1[...,1], marker="o", markersize=4, linestyle="dotted", color=colors[p1])
					plt.title("{}V-{}-{}".format(args.best_k, args.weight_sim, args.num_traj))
					plt.savefig("plots/{}/{}-{}/batch{}_traj{}.png".format(args.dset_name, args.weight_sim, args.best_k, b, t))
					plt.close()
			else:
				for p1 in range(num_ped):
					seq_p1 = seq_b[p1,...].squeeze(0) 
					gt_p1 = target_b[p1,...].squeeze(0)  
					traj_p1 = traj[:,p1,...].squeeze(0) 
					gt_p1 = torch.cat((seq_p1[...,-1,:].unsqueeze(0),gt_p1),dim=0)  
					
					for p2 in range(p1+1, num_ped):  
						print(f"Plotting trajectories for pedestrians {p1} and {p2}")
						seq_p2 = seq_b[p2,...].squeeze(0)   
						gt_p2 = target_b[p2,...].squeeze(0)   
						colors = plt.cm.rainbow(np.linspace(0, 1, 2))  
						traj_p2 = traj[:,p2,...].squeeze(0) 
						gt_p2 = torch.cat((seq_p2[...,-1,:].unsqueeze(0),gt_p2),dim=0)  
						xmax = max([max(15*gt_p1[...,0]), max(15*gt_p2[...,0]), max(15*seq_p1[...,0]), max(15*seq_p2[...,0])])+1
						xmin = min([min(15*gt_p1[...,0]), min(15*gt_p2[...,0]), min(15*seq_p1[...,0]), min(15*seq_p2[...,0])])-1
						ymax = max([max(15*gt_p1[...,1]), max(15*gt_p2[...,1]), max(15*seq_p1[...,1]), max(15*seq_p2[...,1])])+1
						ymin = min([min(15*gt_p1[...,1]), min(15*gt_p2[...,1]), min(15*seq_p1[...,1]), min(15*seq_p2[...,1])])-1
						trajectory_plots = []
						#fig, ax = plt.subplots() 
						for t in range(args.num_traj):
							#print(f"Plotting trajectory {t}")    
							image_array = []
							pred1 = traj_p1[t,...]
							pred2 = traj_p2[t,...]
							pred1 = torch.cat((seq_p1[...,-1,:].unsqueeze(0), pred1), dim=0)
							pred2 = torch.cat((seq_p2[...,-1,:].unsqueeze(0), pred2), dim=0)
							fig, ax = plt.subplots()
							ax.plot(15*seq_p1[...,0], 15*seq_p1[...,1], marker="o", markersize=4,color=colors[0])
							ax.plot(15*seq_p2[...,0], 15*seq_p2[...,1], marker="o", markersize=4,color=colors[1])
							ax.plot(15*pred1[...,0], 15*pred1[...,1], marker="o", markersize=4, linestyle="dotted", color=colors[0])
							ax.plot(15*pred2[...,0], 15*pred2[...,1], marker="o", markersize=4, linestyle = "dotted", color=colors[1])
							plt.xlim([xmin, xmax])
							plt.ylim([ymin, ymax])
							plt.title("{}V-{}-{}".format(args.best_k, args.weight_sim, args.num_traj))
							plt.savefig("plots/{}/{}-{}/batch{}_p1{}_p2{}_traj{}.png".format(args.dset_name, args.weight_sim, args.best_k, b,p1, p2, t))
							plt.close()
							"""
							for s in range(args.obs_len): #uence_args.seqlength):
								fig = plt.figure()
								sns.lineplot(15*seq_p1[...,:s,0],15*seq_p1[...,:s,1], marker="o", color=colors[0])
								sns.lineplot(15*seq_p2[...,:s,0], 15*seq_p2[...,:s,1], marker="o", color=colors[1])
								plt.title("Observation Time Window: $t_{obs}$ %d" %(s))
								plt.xlim([xmin, xmax])
								plt.ylim([ymin, ymax])
								fig.canvas.draw()
								image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8') 
								image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 
								plt.close()
								image_array+=[image]
							for p in range(args.pred_len):
								fig = plt.figure()
								sns.lineplot(15*seq_p1[...,:s,0],15*seq_p1[...,:s,1], marker="o", color=colors[0])
								sns.lineplot(15*seq_p2[...,:s,0], 15*seq_p2[...,:s,1], marker="o", color=colors[1])
								sns.lineplot(15*pred1[...,:p+1,0], 15*pred1[...,:p+1,1], marker="o", color=colors[0])
								sns.lineplot(15*pred2[...,:p+1,0], 15*pred2[...,:p+1,1], marker="o", color=colors[1])
								plt.title("Prediction Time Window: $t_{pred}$ %d" %(p))
								plt.xlim([xmin, xmax])
								plt.ylim([ymin, ymax])
								fig.canvas.draw()
								image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
								image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
								plt.close()   
								image_array+=[image]   
							imageio.mimsave("plots/{}/{}-{}/batch{}_p1{}_p2{}_traj{}.gif".format(args.dset_name, args.weight_sim, args.best_k, b,p1, p2, t), image_array, fps=1.5)
							
							"""


