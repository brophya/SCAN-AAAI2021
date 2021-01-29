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
if args.gpu_id: torch.cuda.set_device(args.gpu_id)

print("\nProcessing Test Data")
testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
print(f"Number of Test Samples: {len(testdataset)}\nNumber of Pedestrians per Sample: {min(testdataset.pedestrian_count.values())} - {max(testdataset.pedestrian_count.values())}")

testloader = DataLoader(testdataset, batch_size=args.eval_batch_size if not args.eval_batch_size is None else len(testdataset), collate_fn=collate_function(), shuffle=False)

generator = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_type=args.domain_type, domain_parameter=args.domain_parameter, delta_bearing=30, delta_heading=30, pretrained_scene="resnet18", device=device, noise_dim=args.noise_dim, noise_type="gaussian").float().to(device)
discriminator=TrajectoryDiscriminator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len,feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,attention_dim=args.attention_dim, domain_type=args.domain_type, domain_parameter=args.domain_parameter,delta_bearing=30, delta_heading=30).float().to(device)
#discriminator.spatial_attn.domain.requires_grad_(False)

best_loss=float(1000)

logger = {}
for key in ['encoder_dim', 'decoder_dim', 'attention_dim', 'embedding_dim', 'domain_type', 'domain_parameter']: logger[key] = getattr(args, key)

model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.best_k}-{args.weight_sim}"
g_file = f"{model_file}_g"
d_file=f"{model_file}_d"

print("---- LOADING SAVED MODEL PARAMETER ----")
generator.load_state_dict(torch.load(g_file+".pt"))
discriminator.load_state_dict(torch.load(d_file+".pt"))

print("---- EVALUATING MODEL ----")
test_ade, test_fde = check_accuracy(testloader, generator, discriminator, args.num_traj) 
print(f"Test ADE: {test_ade:.3f} Test FDE: {test_fde:.3f}")





    
        


