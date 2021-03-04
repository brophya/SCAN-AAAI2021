from __future__ import print_function

import os
import sys
sys.dont_write_bytecode=True

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt 

import glob
import torch
import csv
import pandas as pd 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from termcolor import colored

from arguments import *
from model import *
from data import *
from generative_utils import *

args = parse_arguments()

seed = 100
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.initial_seed()
torch.set_printoptions(precision=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_id = get_free_gpu().item()
torch.cuda.set_device(gpu_id)


if not args.test_only:
	print("TRAINING DATA")
	traindataset = dataset(glob.glob(f"data/{args.dset_name}/train/*.txt"), args)
	print(f"Number of Training Samples: {len(traindataset)}")
	print("VALIDATION DATA")
	valdataset = dataset(glob.glob(f"data/{args.dset_name}/val/*.txt"), args)
	print(f" Number of Validation Samples: {len(valdataset)}")

print("TEST DATA")
testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
print(f"Number of Test Samples: {len(testdataset)}")

print("-"*100)

if not args.test_only:
	trainloader = DataLoader(traindataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True)
	validloader = DataLoader(valdataset, batch_size=args.eval_batch_size if not args.eval_batch_size is None else len(valdataset), collate_fn=collate_function(), shuffle=False)
	testloader = DataLoader(testdataset, batch_size=args.eval_batch_size if not args.eval_batch_size is None else len(testdataset), collate_fn=collate_function(), shuffle=False)

print("---- Model Summary ---->")
generator = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim,  domain_parameter=args.domain_parameter, delta_bearing=30, delta_heading=30, pretrained_scene="resnet18", device=device, noise_dim=args.noise_dim, noise_type=args.noise_type).float().to(device)
discriminator=TrajectoryDiscriminator(seq_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, hidden_size=args.encoder_dim, mlp_dim=1024, attention_dim=args.attention_dim, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, domain_parameter=args.domain_parameter)
discriminator=discriminator.float().to(device)

print("Generator ->", generator)
print("Discriminator ->", discriminator)

if not args.test_only:
	if hasattr(generator, 'spatial_attention'):
		opt_g = torch.optim.Adam([{'params': [p[1] for p in list(generator.named_parameters()) if not 'domain' in p[0]], 'lr': args.lr_g},
				{'params': generator.spatial_attention.domain, 'lr': 5*args.lr_g}])

	else:
		opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g) #, weight_decay=0.1)

	opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d) #, weight_decay=5e-02) #5e-02)

	if args.scheduler:
		sch_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, threshold=0.01, patience=15, factor=0.5, verbose=True)
		sch_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, threshold=0.01, patience=15, factor=0.5, verbose=True)


	best_loss=float(1000)


	logger = {}
	logger['seed']=seed
	for key in ['dset_name', 'model_type', 'obs_len', 'pred_len', 'encoder_dim', 'decoder_dim', 'attention_dim', 'embedding_dim',  'domain_parameter', 'best_k', 'num_traj', 'weight_sim']: logger[key] = getattr(args, key)
	print("-"*31)
	print("|       MODEL PARAMETERS       |")
	print("-"*31)
	for key, value in logger.items(): print(f"{key}:\t{value}")
	print("-"*31)


	train_loss, val_loss, test_loss = [], [], []
	generator.apply(init_weights)
	discriminator.apply(init_weights)


model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.best_k}-{args.weight_sim}"

if args.discriminator_spatial: model_file=f"{model_file}_spatial_d"

if not os.path.exists(f"./trained-models/{args.model_type}/{args.dset_name}"): 
	print(f"Creating directory ./trained-models/{args.model_type}/{args.dset_name}")
	os.makedirs(f"./trained-models/{args.model_type}/{args.dset_name}")

g_file = f"{model_file}_g"
d_file=f"{model_file}_d"

if args.train_saved:
	generator.load_state_dict(torch.load(f"{g_file}.pt"))
	discriminator.load_state_dict(torch.load(f"{d_file}.pt"))


if args.test_only:
	print("Evaluating Trained Model")
	generator.load_state_dict(torch.load(f"{g_file}.pt"))
	discriminator.load_state_dict(torch.load(f"{d_file}.pt"))
	testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
	test_ade, test_fde = check_accuracy(testloader, generator, discriminator, args.num_traj)
	print(f"Test ADE: {test_ade.item():.3f}")
	print(f"Test FDE: {test_fde.item():.3f}")
	exit()


# generator trained x num_iter times discriminator trained 
num_iter=1 
logger['num_iter']=num_iter

early_stopping=EarlyStopping()

print("---- TRAINING ---->")

for epoch in range(args.num_epochs):
	print("*"*20)
	epoch_ade = float(0)
	generator.train()
	discriminator.train()
	pbar_train = tqdm(range(len(trainloader)), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
	for b, batch in enumerate(trainloader):
	
		if (b%num_iter)==0: 
			loss_d = discriminator_step(b, batch, generator, discriminator, opt_d, discriminator_spatial=args.discriminator_spatial)

		loss_g, ade, fde  = generator_step(b, batch, generator, discriminator=discriminator, optimizer_g=opt_g, best_k=args.best_k, weight_sim=args.weight_sim, train=True, discriminator_spatial=args.discriminator_spatial, l2_loss_weight=args.l2_loss_weight)
		pbar_train.update(1)  
		pbar_train.set_description(f"[Epoch: {epoch+1}/{args.num_epochs}] Batch: {b+1}/{len(trainloader)} ADE: {ade:.3f}")
		epoch_ade+=ade.item()
	epoch_ade/=(b+1)
	train_loss+=[epoch_ade]
	pbar_train.set_description(f"[Epoch: {epoch+1}/{args.num_epochs}] Train ADE: {epoch_ade:.3f}")
	pbar_train.close()
	generator.eval()
	discriminator.eval()
	val_ade, valid_fde = check_accuracy(validloader, generator, discriminator, args.num_traj) 
	print(f"[Epoch: {epoch+1}/{args.num_epochs}] Valid ADE: {val_ade:.3f}")
	test_ade, test_fde = check_accuracy(testloader, generator, discriminator, args.num_traj) 
	if args.scheduler:
		sch_g.step(val_ade)
		sch_d.step(val_ade)

	if (val_ade<best_loss):
		best_loss=val_ade 
		logger['ade'] = test_ade.item()
		logger['fde'] = test_fde.item()
		torch.save(generator.state_dict(), f"{g_file}.pt")
		torch.save(discriminator.state_dict(), f"{d_file}.pt")
	

	early_stopping(val_ade) 
	if early_stopping.early_stop:
		print("early stopping..")
		print(f"[Epoch: {epoch+1}/{args.num_epochs}] Test ADE: {logger['ade']:.3f} Test FDE: {logger['fde']:.3f}")
		break 

	val_loss+=[val_ade]
	test_loss+=[test_ade]
	print(f"[Epoch: {epoch+1}/{args.num_epochs}] Test ADE: {logger['ade']:.3f} Test FDE: {logger['fde']:.3f}")
	
	if (epoch+1)%10 ==0:
		if hasattr(generator, 'spatial_attention') and hasattr(generator.spatial_attention, 'domain'): print(generator.spatial_attention.domain.data)

print("Finished Training")

print("Evaluating Trained Model")
generator.load_state_dict(torch.load(f"{g_file}.pt"))
discriminator.load_state_dict(torch.load(f"{d_file}.pt"))
testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
test_ade, test_fde = check_accuracy(testloader, generator, discriminator, args.num_traj)
logger['ade'] = test_ade.item()
logger['fde'] = test_fde.item() 
print(f"Test ADE: {test_ade.item():.3f}")
print(f"Test FDE: {test_fde.item():.3f}")

if args.plot_curves:
	print("Saving training curves")
	fig = plt.figure()
	plt.plot(range(len(train_loss)), train_loss, label="train ade")
	plt.plot(range(len(val_loss)), val_loss, label="valid ade")
	plt.plot(range(len(test_loss)), test_loss, label="test ade")
	plt.legend()
	plt.title("ADE: {:.3f}\nFDE: {:.3f}".format(logger['ade'], logger['fde']))
	plt.savefig(f"{model_file}.png")
	plt.close()

	print("Plotting Domain")
	if hasattr(generator, 'spatial_attention') and hasattr(generator.spatial_attention, 'domain'):
	    plot_file = f"{g_file}-domain.png"
	    plot_domain(generator, plot_file, args.delta_heading, args.delta_bearing)


print("logging Results")
writeheader=True
if os.path.exists('logger_generative.csv'): writeheader=False

f = open('logger_generative.csv', 'a')
w = csv.DictWriter(f, logger.keys())
if writeheader: w.writeheader()
w.writerow(logger)
f.close()

print("Results written to File")

train_logger = f"{args.model_type}-{args.dset_name}-{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.delta_bearing}-{args.delta_heading}-{args.obs_len}-{args.pred_len}-{args.best_k}-{args.weight_sim}-train.csv"
np.savetxt("results/{}".format(train_logger), train_loss, delimiter=",")
print("Results written to results/{}".format(train_logger))










    
        


