from __future__ import print_function

import os
import sys

sys.dont_write_bytecode = True

import warnings

warnings.filterwarnings("ignore")

import glob
import time
import torch
import random
import tensorflow as tf

from torch.utils.data import DataLoader

from arguments import parse_arguments
from model import TrajectoryGenerator, TrajectoryDiscriminator
from data import dataset, collate_function
from generative_utils import discriminator_step, generator_step, check_accuracy

from utils import *

args = parse_arguments()

print(args.__dict__)

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.initial_seed()
torch.set_printoptions(precision=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	#gpu_id = get_free_gpu().item()
	torch.cuda.set_device(0)

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
print("-" * 100)

if not args.test_only:
	trainloader = DataLoader(traindataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True)
	validloader = DataLoader(valdataset,
							 batch_size=args.eval_batch_size if not args.eval_batch_size is None else len(valdataset),
							 collate_fn=collate_function(), shuffle=False)
	testloader = DataLoader(testdataset,
							batch_size=args.eval_batch_size if not args.eval_batch_size is None else len(testdataset),
							collate_fn=collate_function(), shuffle=False)

generator = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2,
								embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim,
								decoder_dim=args.decoder_dim, attention_dim=args.attention_dim,
								domain_parameter=args.domain_parameter, delta_bearing=30, delta_heading=30,
								pretrained_scene="resnet18", device=device, noise_dim=args.noise_dim,
								noise_type=args.noise_type, noise_mix_type=args.noise_mix_type).float().to(device)
discriminator = TrajectoryDiscriminator(args.d_model_type, seq_len=(
			args.obs_len + args.pred_len) if args.d_type == 'global' else args.pred_len, feature_dim=2,
										embedding_dim=args.embedding_dim, hidden_size=args.encoder_dim, mlp_dim=1024,
										attention_dim=args.attention_dim, delta_bearing=args.delta_bearing,
										delta_heading=args.delta_heading, domain_parameter=args.domain_parameter)

discriminator = discriminator.float().to(device)

if not args.test_only:
	opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)

	opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)

	if args.scheduler:
		sch_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, threshold=0.01, patience=10, factor=0.5, verbose=True,
														   min_lr=1e-04)
		sch_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, threshold=0.01, patience=10, factor=0.5, verbose=True,
														   min_lr=1e-04)

	best_loss = float(1000)
	generator.apply(init_weights)
	discriminator.apply(init_weights)

model_file = f"trained-models/generative_spatial_temporal/{args.dset_name}/{args.best_k}V-{args.l}"

if not os.path.exists(f"./trained-models/{args.model_type}/{args.dset_name}"):
	print(f"Creating directory ./trained-models/{args.model_type}/{args.dset_name}")
	os.makedirs(f"./trained-models/{args.model_type}/{args.dset_name}")

g_file = f"{model_file}_g"
d_file = f"{model_file}_d"

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

print("---- TRAINING ---->")

for epoch in range(args.num_epochs):
	print("*" * 20)
	epoch_ade = float(0)
	d_loss_ = float(0)
	g_loss_ = float(0)
	generator.train()
	discriminator.train()
	epoch_time = time.time()
	for b, batch in enumerate(trainloader):
		loss_d = discriminator_step(b, batch, generator, discriminator, opt_d, d_spatial=args.d_spatial,
									d_type=args.d_type, d_domain=args.d_domain)
		d_loss_ += loss_d.item()
		loss_g, ade, fde, generator_pred, pedestrians, _ = generator_step(b, batch, generator,
																		  discriminator=discriminator,
																		  optimizer_g=opt_g, best_k=args.best_k,
																		  l=args.l, train=True,
																		  d_spatial=args.d_spatial,
																		  l2_loss_weight=args.l2_loss_weight,
																		  clip=args.clip, d_type=args.d_type,
																		  d_domain=args.d_domain)
		epoch_ade += ade.item()
		g_loss_ += loss_g.item()
	epoch_time = time.time() - epoch_time
	d_loss_ /= (b + 1)
	g_loss_ /= (b + 1)
	epoch_ade /= (b + 1)
	print(
		f"[Epoch: {epoch + 1}/{args.num_epochs}] Train ADE: (Min over {args.best_k}): {epoch_ade:.3f} Loss_G: {g_loss_:.3f} Loss_D: {d_loss_:.3f} --- Time Per Epoch: {epoch_time:.3f}")
	generator.eval()
	discriminator.eval()
	val_ade, valid_fde, val_ade_ = check_accuracy(validloader, generator, discriminator, args.num_traj)
	print(
		f"[Epoch: {epoch + 1}/{args.num_epochs}] Valid ADE (Min over {args.num_traj}): {val_ade:.3f} (Min over 1): {val_ade_:.3f}")
	if args.scheduler:
		sch_g.step(val_ade)
		sch_d.step(val_ade)
	if (val_ade < best_loss):
		best_loss = val_ade
		test_ade, test_fde, test_ade_ = check_accuracy(testloader, generator, discriminator, args.num_traj)
		torch.save(generator.state_dict(), f"{g_file}.pt")
		torch.save(discriminator.state_dict(), f"{d_file}.pt")
	print(
		f"[Epoch: {epoch + 1}/{args.num_epochs}] Test ADE (Min over {args.num_traj}): {test_ade.item():.3f} (Min over 1): {test_ade_:.3f} Test FDE: {test_fde.item():.3f}")

print("Finished Training")

print("Evaluating Trained Model")
generator.load_state_dict(torch.load(f"{g_file}.pt"))
discriminator.load_state_dict(torch.load(f"{d_file}.pt"))
testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
_, test_fde, test_ade = check_accuracy(testloader, generator, discriminator, args.num_traj)
print(f"Test ADE: {test_ade.item():.3f}")
print(f"Test FDE: {test_fde.item():.3f}")




