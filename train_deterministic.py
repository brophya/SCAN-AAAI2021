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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
from tqdm import tqdm 
from arguments import *
from model import *
from data import *

args = parse_arguments()

seed = 30
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
	logger = {}
	logger['seed']=seed
	for key in ['dset_name', 'obs_len', 'pred_len', 'model_type', \
		'encoder_dim', 'decoder_dim', 'attention_dim', 'embedding_dim', \
		 'domain_parameter', 'delta_bearing', 'delta_heading']: 
		logger[key] = getattr(args, key)

	print("-"*31)
	print("|       TRAIN PARAMETERS       |")
	print("-"*31)
	for key, value in logger.items(): print(f"{key}:\t{value}")
	print("-"*31)

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

model = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_parameter=args.domain_parameter, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, pretrained_scene="resnet18", device=device, noise_dim=None, noise_type=None).float().to(device)

model.apply(init_weights)

if hasattr(model, 'spatial_attention'):
	optimizer = torch.optim.Adam([{'params': [p[1] for p in list(model.named_parameters()) if not 'domain' in p[0]], 'lr': args.lr},
				{'params': model.spatial_attention.domain, 'lr': 5*args.lr}])

else:
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=8e-02)
if args.scheduler:
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.01, patience=15, factor=0.5, verbose=True)

best_loss=float(1000)

model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.delta_bearing}-{args.delta_heading}-{args.obs_len}-{args.pred_len}"

if not os.path.exists(f"./trained-models/{args.model_type}/{args.dset_name}"): 
	print(f"Creating directory ./trained-models/{args.model_type}/{args.dset_name}")
	os.makedirs(f"./trained-models/{args.model_type}/{args.dset_name}")

if args.train_saved:
	model.load_state_dict(torch.load(f"{model_file}.pt"))

if args.test_only:
	print("Evaluating trained model")
	model.load_state_dict(torch.load(f"{model_file}.pt"))
	testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
	test_ade, test_fde = evaluate_model(model, testloader)
	print(f"Test ADE: {test_ade:.3f}")
	print(f"Test FDE: {test_fde:.3f}")
	exit()

print("TRAINING")

train_loss=[]
val_loss=[]
test_loss=[]

early_stopping=EarlyStopping()

for epoch in range(args.num_epochs):
	epoch_ade = float(0)
	model.train()
	pbar_train = tqdm(range(len(trainloader)), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') 
	for b, batch in enumerate(trainloader):
		optimizer.zero_grad()
		pred, target, sequence, pedestrians = predict(batch, model)
		ade_b, fde_b = eval_metrics(pred, target, pedestrians)
		pbar_train.update(1)
		pbar_train.set_description(f"EPOCH: {epoch+1} ade: {ade_b:.3f}")  
		ade_b.backward()
		optimizer.step()
		epoch_ade+=ade_b.item()
	epoch_ade/=(b+1)
	pbar_train.set_description(f"EPOCH: {epoch+1} Train ADE: {epoch_ade:.3f}")
	pbar_train.close()
	train_loss+=[epoch_ade]
	model.eval()
	val_ade, valid_fde = evaluate_model(model, validloader)
	if args.scheduler:
		scheduler.step(val_ade)
	if (val_ade<best_loss):
		best_loss=val_ade 
		torch.save(model.state_dict(), f"{model_file}.pt")
		test_ade, test_fde = evaluate_model(model, testloader)
		logger['ade'] = test_ade
		logger['fde'] = test_fde 
	early_stopping(val_ade) 
	if early_stopping.early_stop:
		print("early stopping..")
		break 
	val_loss+=[val_ade]
	test_loss+=[test_ade]
	print(f"Valid ADE: {val_ade:.3f}\nTest ADE: {logger['ade']:.3f} Test FDE: {logger['fde']:.3f}")  
	if (epoch+1)%10 ==0 and hasattr(model, 'spatial_attention') and hasattr(model.spatial_attention, 'domain'): print(model.spatial_attention.domain.data)
	print("-"*20)

print("Finished Training")

model.eval()

print("Evaluating trained model")
model.load_state_dict(torch.load(f"{model_file}.pt"))
testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
test_ade, test_fde = evaluate_model(model, testloader)
print(f"Test ADE: {test_ade:.3f}")
print(f"Test FDE: {test_fde:.3f}")
logger['ade'] = test_ade
logger['fde'] = test_fde

args.plot_curves=False
if args.plot_curves:
	print("Plotting training curves")
	fig = plt.figure()
	plt.plot(range(len(train_loss)), train_loss, label="train ade")
	plt.plot(range(len(val_loss)), val_loss, label="valid ade")
	plt.plot(range(len(test_loss)), test_loss, label="test ade")
	plt.legend()
	plt.title("ADE: {:.3f}\nFDE: {:.3f}".format(logger['ade'], logger['fde']))
	plt.savefig(f"{model_file}.png")
	plt.close()


	if hasattr(model, 'spatial_attention') and hasattr(model.spatial_attention,'domain'):
		print("Plotting learned spatial domain")
		plot_file = f"{model_file}-domain.png"
		plot_domain(model, plot_file, args.delta_heading, args.delta_bearing)


print("Logging Results")
writeheader=True
if os.path.exists('logger.csv'): writeheader=False

f = open('logger.csv', 'a')
w = csv.DictWriter(f, logger.keys())
if writeheader: w.writeheader()
w.writerow(logger)
f.close()

print("Results written to File")

train_logger = f"{args.model_type}-{args.dset_name}-{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.delta_bearing}-{args.delta_heading}-{args.obs_len}-{args.pred_len}-train.csv"
np.savetxt("results/{}".format(train_logger), train_loss, delimiter=",")
print("Results written to results/{}".format(train_logger))

print("-"*30)




    
        


