import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import imageio

import glob

import matplotlib
import seaborn as sns
matplotlib.use("agg")
from matplotlib import pyplot as plt

from model import *
from generative_utils import *
from utils import *
from arguments import *
from data import *

seed = 12 #12
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.initial_seed()
torch.set_printoptions(precision=2)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['lines.linewidth'] = 2.0

def get_df(traj_p):
	df = []
	columns = ['ix', 'x', 'y']
	#columns = ['ix', 't', 'x', 'y']
	for ix in range(traj_p.size(0)):
		#for t in range(traj_p.size(1)):
			#df.append([ix, t, traj_p[ix, t, 0].item(), traj_p[ix, t, 1].item()])
		df.append([ix, traj_p[ix,...,0].item(), traj_p[ix,...,1].item()])
	df=pd.DataFrame(df, columns=columns)
	return df

def plot_pedestrian(seq, pred, c):
	# trajectories x prediction_length x 2 
	mean_traj = torch.cat((seq[...,-1,:].unsqueeze(0), pred.mean(dim=0)), dim=0)	
	sns.lineplot(seq[...,0], seq[...,1],marker="o", markersize=3, linewidth=2.0, color='red')
	#traj = torch.cat((seq[...,-1,:].unsqueeze(0).repeat(pred.shape[0], 1, 1), pred), dim=1)
	print(pred.size())
	df = get_df(pred[:,-1,:])
	#for i in range(traj.shape[0]):
	#	sns.lineplot(traj[i,...,0], traj[i,...,1], color="blue", linestyle="--", linewidth=0.8, zorder=1)
	sns.kdeplot(df['x'], df['y'],shade=False, shade_lowest=False, color="white")	
	#sns.kdeplot(data=df, x='x', y='y', fill=True, alpha=1) #, thresh=0.0001)
	#plt.scatter(mean_traj[...,0], mean_traj[...,1], marker="x", color=c, s=5)
	sns.lineplot(mean_traj[...,0], mean_traj[...,1], marker="o", linewidth=2.0, markersize=3, color='red', zorder=2)

def get_prediction(batch, generator, args):
	batch = get_batch(batch)
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, \
	op_mask,pedestrians, scene_context, batch_mean, batch_var = batch
	predictions, target, sequence, pedestrians = predict_multiple(batch, generator, args.num_traj)
	return predictions, sequence	

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments()

testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
print(f"Number of Test Samples: {len(testdataset)}")

print("-"*100)

testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
k=args.best_k
l=args.l

generator = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim,  domain_parameter=args.domain_parameter, delta_bearing=30, delta_heading=30, pretrained_scene="resnet18", device=device, noise_dim=args.noise_dim, noise_type=args.noise_type, noise_mix_type=args.noise_mix_type).float().to(device)
model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.best_k}V-{args.l}"
g_file = f"{model_file}_g.pt"

generator.load_state_dict(torch.load(g_file))

dirname = f"plots/{k}-{l}-{args.dset_name}"
if not os.path.exists(dirname): os.makedirs(dirname)


img_array=[]
for b, batch in enumerate(testloader):
	if (b+1)==224:
		imageio.mimwrite(f"plots/{k}-{l}-{args.dset_name}/movie.gif", img_array, fps=2)
		exit()
	print(f"Plotting density plots for batch {b+1}/{len(testloader)}")
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask, \
	op_mask,pedestrians, scene_context, batch_mean, batch_var = batch
	if pedestrians.data<2:
		continue
	predictions, sequence = get_prediction(batch, generator, args)	
	predictions = predictions.squeeze()
	predictions = predictions.clone().detach().cpu()
	sequence = sequence.squeeze(0).clone().detach().cpu()
	target = target.squeeze(0).clone().detach().cpu()
	gt_traj = torch.cat((sequence, target), dim=1)
	xlim = [gt_traj[...,0].min()-1.0, gt_traj[...,0].max()+1.0]
	ylim = [gt_traj[...,1].min()-1.0, gt_traj[...,1].max()+1.0]
	num_ped, slen = sequence.size()[:2] 
	num_ped=2
	colors = plt.cm.tab10(np.linspace(0,1,num_ped))
	print(f"Number of pedestrians: {num_ped}")
	#fig = plt.figure()
	#img = plt.imread(f"zara01/img-{b+1}.png")
	#plt.imshow(img, alpha=0.6, extent = [0, 16, 0, 14], zorder=0)
	img_array=[]
	for p in range(target.size(1)):
		fig=plt.figure()
		pred = predictions[:, :, :p+1, ...]
		for p1 in range(num_ped):
			if (p1%2)==0:
				continue
			seq_p1 = sequence[p1,...]
			pred_p1 = pred[:,p1,...]
			plot_pedestrian(seq_p1, pred_p1, colors[p1])
		plt.title(f"${k}V-{l}$", fontsize=15)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel(' ')
		plt.ylabel(' ')
		plt.tight_layout()
		#plt.xlim([0, 16])
	
		#plt.ylim([0, 14])
		plt.savefig(f"plots/{k}-{l}-{args.dset_name}/{b+1}.png") #-{p1}-{p2}.png") #-{p3}.png")
		data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		img_array+=[data] 
	imageio.mimwrite(f"plots/{k}-{l}-{args.dset_name}/{b+1}.gif", img_array, fps=2.5) 
