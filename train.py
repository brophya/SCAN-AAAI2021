from __future__ import print_function

import sys
sys.dont_write_bytecode=True

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import glob
import torch
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from arguments import *
from model import *
from data import *
from metrics import *
from evaluate import *
from generative_utils import *
from argoverse_data import *

args = parse_arguments()

seed=100
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.initial_seed()
torch.manual_seed(seed)
torch.backends.cudnn.deterministic=True

torch.set_printoptions(precision=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.gpu_id:
	torch.cuda.set_device(args.gpu_id)

if not 'argoverse' in args.dset_name:
	args.obs_len = 8
	args.pred_len = 12

if not 'argoverse' in args.dset_name:
	traindataset = dataset(glob.glob("data/{}/train/*.txt".format(args.dset_name)), args)
	testdataset = dataset(glob.glob("data/{}/test/*.txt".format(args.dset_name)), args)
	valdataset = dataset(glob.glob("data/{}/val/*.txt".format(args.dset_name)), args)
	trainloader = DataLoader(traindataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True, num_workers=4)
	validloader = DataLoader(valdataset, batch_size=len(valdataset), collate_fn=collate_function(), shuffle=False) #, num_workers=4)
	testloader = DataLoader(testdataset, batch_size=len(testdataset), collate_fn=collate_function(), shuffle=False) #, num_workers=4)
else:
	# ArgoverseData
	traindataset = ArgoverseData(glob.glob("/bigtemp/js3cn/argoverse/train/data/*.csv")[:1000], args)
	valdataset = ArgoverseData(glob.glob("/bigtemp/js3cn/argoverse/val/data/*.csv")[:1000], args)
	#testdataset = ArgoverseData(glob.glob("/bigtemp/js3cn/argoverse/test_obs/data/*.csv")[:1000], args)
	trainloader = DataLoader(traindataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True, num_workers=4)
	validloader = DataLoader(valdataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=False, num_workers=4)
	#testloader = DataLoader(valdataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=False, num_workers=4)

maxPeds = max([traindataset.maxPedestrians, testdataset.maxPedestrians, valdataset.maxPedestrians])
print(maxPeds*maxPeds)
if 'generative' in args.model_type:
	generator = TrajectoryGenerator(args.model_type, args.obs_len, args.pred_len, args.ip_dim, args.op_dim, args.embedding_dim, args.encoder_dim, args.decoder_dim, args.attention_dim, device, args.domain_type, args.param_domain, args.delta_bearing, args.delta_heading, args.domain_init_type, noise_dim=args.noise_dim, noise_type=args.noise_type)
	discriminator = TrajectoryDiscriminator(args.obs_len, args.pred_len, args.embedding_dim, args.encoder_dim, args.delta_bearing, args.delta_heading, args.attention_dim, device=device, domain_type=args.domain_type, param_domain=args.param_domain, domain_init_type=args.domain_init_type)
	generator = generator.float().to(device)
	discriminator = discriminator.float().to(device)
	print("Model Summary-->")
	print("Generator-->")
	print(generator)
	print("Discriminator-->")
	print(discriminator)
	g_param = [p for p in generator.parameters() if p.requires_grad]
	d_param = [p for p in discriminator.parameters() if p.requires_grad]
	optimizer_g = torch.optim.Adam(g_param, lr=args.lr)
	optimizer_d = torch.optim.Adam(d_param, lr=args.lr)
else:
	model = TrajectoryGenerator(args.model_type, args.obs_len, args.pred_len, args.ip_dim, args.op_dim, args.embedding_dim, args.encoder_dim, args.decoder_dim, args.attention_dim, device, args.domain_type, args.param_domain, args.delta_bearing, args.delta_heading, args.domain_init_type, maxPeds=maxPeds)
	model = model.float().to(device)
	print("Model Summary->")
	print(model)
	if hasattr(model, 'spatial_attn') and not (args.domain_type=='learnable'): 
	   model.spatial_attn.domain.requires_grad_(False)
	   params = [p for p in model.parameters() if p.requires_grad]
	   optimizer = torch.optim.Adam(params, lr=args.lr)
	else:	
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	
best_loss=float(1000)
if args.log_results:
	log_file = "trained-models/{}/{}/log.csv".format(args.model_type, args.dset_name)
	columns = ['seed', 'encoder_dim', 'decoder_dim', 'embedding_dim', 'attention_dim']
	current_cols = [seed, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim]
	metric_columns = ['test_ade', 'test_fde']
	if ('spatial' in args.model_type):
		spatial_columns = ['domain_type', 'domain_init_type', 'domain_init']
		current_cols.extend([args.domain_type, args.domain_init_type, args.param_domain])
		columns.extend(spatial_columns)
	if ('generative' in args.model_type):
		columns.extend(['best_k', 'diversity_loss_weight'])
		current_cols.extend([args.best_k, args.weight_sim])
	columns.extend(metric_columns)
	print(columns)
	print(current_cols)
	current_cols.extend([best_loss, best_loss])
	current_cols = np.array(current_cols).reshape(1,len(current_cols))
	if not os.path.exists(log_file):
		log_df = pd.DataFrame(current_cols, columns=columns)
	else:
		log_df = pd.read_csv(log_file, names=columns)
	saved_df = current_cols[:-2]


if not 'generative' in args.model_type:
	model_file = "./trained-models/{}/{}/{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim)
	if not os.path.exists("./trained-models/{}/{}/".format(args.model_type, args.dset_name)):
		os.makedirs("./trained-models/{}/{}/".format(args.model_type, args.dset_name))
	if 'spatial' in args.model_type:
		model_file = "{}_{}_{}_{}".format(model_file, args.domain_type, args.domain_init_type, args.param_domain)
else:
	g_file = "./trained-models/{}/{}/g_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim ,args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
	d_file = "./trained-models/{}/{}/d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim, args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
	if not os.path.exists("./trained-models/{}/{}".format(args.model_type, args.dset_name)):
		os.makedirs("./trained-models/{}/{}".format(args.model_type, args.dset_name))
	
if not 'generative' in args.model_type:
	train_loss = []
	val_loss = []
	test_loss = []
	for epoch in range(args.num_epochs):
		epoch_ade = float(0)
		model.train()
		for b, batch in enumerate(trainloader):
			optimizer.zero_grad()
			start_time = time.time()
			pred, target,sequence, pedestrians, agent_idx = predict(batch,model)
			if ('argoverse' in args.dset_name):
				ade_b = ade(pred, target, pedestrians, argoverse=True, agent_idx=agent_idx)
			else:
				ade_b = ade(pred, target, pedestrians)
			epoch_ade+=ade_b.item()
			ade_b.backward()
			optimizer.step()
			if hasattr(model.spatial_attn, 'domain') and ('learnable' in args.domain_type):
					model.spatial_attn.domain.data.clamp_(min=0.0)
		epoch_ade/=(b+1)
		train_loss+=[epoch_ade]
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Train ADE: {epoch_ade:.3f}") 
		model.eval()
		val_ade = float(0)
		val_fde = float(0)
		for b, batch in enumerate(validloader):
			pred, target, sequence, pedestrians, agent_idx = predict(batch, model)
			if ('argoverse' in args.dset_name):
				ade_b = ade(pred, target, pedestrians, argoverse=True, agent_idx=agent_idx)
			else:
				ade_b = ade(pred, target, pedestrians)
			if ('argoverse' in args.dset_name):
				fde_b = fde(pred, target, pedestrians, argoverse=True, agent_idx=agent_idx)
			else:
				fde_b = ade(pred, target, pedestrians)
			val_ade+=ade_b.item()
			val_fde+=fde_b.item()
			del ade_b
		val_ade/=(b+1)
		val_fde/=(b+1)
		val_loss+=[val_ade]
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Valid ADE: {val_ade:.3f}")
		if not 'argoverse' in args.dset_name:
			test_ade, test_fde = evaluate_model(model, testloader)
			test_loss+=[test_ade]
		if hasattr(model.spatial_attn, 'domain') and ('learnable' in args.domain_type) and (epoch%10==0):
				print(model.spatial_attn.domain)
		if (val_ade<best_loss):
			best_loss=val_ade
			best_fde = val_fde
			print("Saving Model..")
			torch.save(model.state_dict(), model_file+".pt")
			if not 'argoverse' in args.dset_name:
				best_ade = test_ade
				best_fde = test_fde
		if not 'argoverse' in args.dset_name:
			print(f"[Epoch: {epoch+1}/{args.num_epochs}] Test ADE: {test_ade:.3f} Test FDE: {test_fde:.3f}")
		print("-"*50)
		fig = plt.figure()
		plt.plot(range(len(train_loss)), train_loss, label="TrainADE")
		plt.plot(range(len(val_loss)), val_loss, label="ValADE")
		if not 'argoverse' in args.dset_name:
			plt.plot(range(len(test_loss)), test_loss, label="TestADE")
		plt.legend()
		if not 'argoverse' in args.dset_name:
			plt.title("ADE: {:.3f}\nFDE: {:.3f}".format(best_ade, best_fde))
		else:
			plt.title("ADE: {:.3f}\nFDE: {:.3f}".format(best_loss, best_fde))
		plt.savefig(model_file+".png")
		plt.close()
	print("Model Training Finished") 
	if hasattr(model.spatial_attn, 'domain') and ('learnable' in args.domain_type):
		print("Plotting learned domain")
		plot_file = model_file+"_domain_{}_{}.png".format(args.delta_heading, args.delta_bearing)
		plot_domain(model, plot_file, args.delta_heading, args.delta_bearing)

	if args.log_results:
		print("Writing Results to File")
		current_cols[:,-2] = best_ade
		current_cols[:,-1] = best_fde
		current_cols = pd.DataFrame(current_cols, columns=columns)
		log_df = log_df.append(current_cols)
		log_df.to_csv(log_file, index=False)

else:
	train_loss = []
	val_loss = []
	test_loss = []
	for epoch in range(args.num_epochs):
		epoch_ade = float(0)  
		generator.train()
		discriminator.train()
		for b, batch in enumerate(trainloader):
			loss_d = discriminator_step(b, batch, generator, discriminator, optimizer_d)
			loss_g, ade, fde  = generator_step(b, batch, generator, discriminator, optimizer_g, args.best_k, args.weight_sim)
			epoch_ade+=ade.item()
			if ('spatial' in args.model_type) and ('learnable' in args.domain_type):
					generator.spatial_attn.domain.data.clamp_(min=0.0)
		epoch_ade/=(b+1)
		train_loss+=[epoch_ade]
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Train ADE: {epoch_ade:.3f}") 
		val_ade, val_fde = check_accuracy(validloader, generator, discriminator, args.num_traj)
		test_ade, test_fde = check_accuracy(testloader, generator, discriminator, args.num_traj)
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Valid ADE: {val_ade:.3f}")
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Test ADE: {test_ade:.3f} Test FDE: {test_fde:.3f}")
		val_loss+=[val_ade]
		test_loss+=[test_ade]
		if (val_ade<best_loss):
			best_loss=val_ade
			best_ade = test_ade
			best_fde = test_fde
			print("Saving Model..")
			torch.save(generator.state_dict(), g_file+".pt")
			torch.save(discriminator.state_dict(), d_file+".pt")
		if ((epoch+1)%10)==0:
			print(generator.spatial_attn.domain.data)
		fig = plt.figure()
		plt.plot(range(len(train_loss)), train_loss, label="TrainADE")
		plt.plot(range(len(val_loss)), val_loss, label="ValADE")
		plt.plot(range(len(test_loss)), test_loss, label="TestADE")
		plt.legend()
		plt.title("ADE: {:.3f}\nFDE: {:.3f}".format(best_ade, best_fde))
		plt.savefig(g_file+".png")
		plt.close()
		print("-"*50)
	print("Model Training Finished")
	print("Plotting learned domain")
	plot_file = g_file+"_domain.png"
	plot_domain(generator, plot_file, args.delta_heading, args.delta_bearing)
	if args.log_results:
		print("Writing Results to File")
		current_cols[:,-2] = best_ade.item()
		current_cols[:,-1] = best_fde.item()
		current_cols = pd.DataFrame(current_cols, columns=columns)
		log_df = log_df.append(current_cols)
		log_df.to_csv(log_file, index=False)

	








