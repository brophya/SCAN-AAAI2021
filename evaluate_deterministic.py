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
from tqdm import tqdm 
from arguments import *
from model import *
from data import *

args = parse_arguments()

seed = 31
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.initial_seed()
torch.set_printoptions(precision=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu_id: torch.cuda.set_device(args.gpu_id)

logger = {}
for key in ['dset_name', 'obs_len', 'pred_len', 'model_type', 'encoder_dim', 'decoder_dim', 'attention_dim', 'embedding_dim', 'domain_type', 'domain_parameter']: logger[key] = getattr(args, key)
print("-"*31)
print("|          PARAMETERS          |")
print("-"*31)
for key, value in logger.items(): print(f"{key}:\t{value}")
print("-"*31)


print("---- Processing Test Data ----")
testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
print(f"Number of Test Samples: {len(testdataset)}\nNumber of Pedestrians per Sample: {min(testdataset.pedestrian_count.values())} - {max(testdataset.pedestrian_count.values())}")
print("-"*31)
testloader = DataLoader(testdataset, batch_size=args.eval_batch_size if not args.eval_batch_size is None else(testdataset), collate_fn=collate_function(), shuffle=False)

model = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_type=args.domain_type, domain_parameter=args.domain_parameter, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, pretrained_scene="resnet18", device=device, noise_dim=None, noise_type=None).float().to(device)

model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.delta_bearing}-{args.delta_heading}-{args.obs_len}-{args.pred_len}"


print(f"---- Loading Saved Parameters from {model_file}.pt -----")
model.load_state_dict(torch.load(model_file+".pt"))
print(f"---- TESTING -----")
model.eval()
test_ade, test_fde = evaluate_model(model, testloader)
print(f"Test ADE: {test_ade:.3f} \tTest FDE: {test_fde:.3f}")




    
        


