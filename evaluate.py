from __future__ import print_function

import sys
sys.dont_write_bytecode=True

import warnings
warnings.filterwarnings("ignore")
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from arguments import *
from model import *
from data import *
from metrics import *
from generative_utils import *

def evaluate_model(model, testloader):
	test_ade = float(0)
	test_fde = float(0)
	model.eval()
	for b, batch in enumerate(testloader):
		pred, target,sequence, pedestrians,_= predict(batch,model)
		ade_b = ade(pred, target, pedestrians)
		fde_b = fde(pred, target, pedestrians)
		test_ade+=ade_b.item()
		test_fde+=fde_b.item()
		del ade_b, fde_b
	test_ade/=(b+1)
	test_fde/=(b+1)
	return test_ade, test_fde

def main(): 
	args = parse_arguments()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.gpu_id:
		torch.cuda.set_device(args.gpu_id)
	if 'generative' in args.model_type:
		generator = TrajectoryGenerator(args.model_type, args.obs_len, args.pred_len, args.ip_dim, args.op_dim, args.embedding_dim, args.encoder_dim, args.decoder_dim, args.attention_dim, device, args.domain_type, args.param_domain, args.delta_bearing, args.delta_heading, args.domain_init_type, noise_dim=args.noise_dim, noise_type=args.noise_type)
		discriminator = TrajectoryDiscriminator(args.obs_len, args.pred_len, args.embedding_dim, args.encoder_dim, args.delta_bearing, args.delta_heading, args.attention_dim, device=device, domain_type=args.domain_type, param_domain=args.param_domain, domain_init_type=args.domain_init_type)
		generator = generator.float().to(device)
		discriminator = discriminator.float().to(device)
	else:
		model = TrajectoryGenerator(args.model_type, args.obs_len, args.pred_len, args.ip_dim, args.op_dim, args.embedding_dim, args.encoder_dim, args.decoder_dim, args.attention_dim, device, args.domain_type, args.param_domain, args.delta_bearing, args.delta_heading, args.domain_init_type, noise_dim=args.noise_dim, noise_type=args.noise_type)
		model = model.float().to(device)
	testdataset = dataset(glob.glob("data/{}/test/*.txt".format(args.dset_name)), args)
	testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
	if not 'generative' in args.model_type:
		model_file = "./trained-models/{}/{}/{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim)
		if 'spatial' in args.model_type:
			model_file = "{}_{}_{}_{}".format(model_file, args.domain_type, args.domain_init_type, args.param_domain)
	else:
		g_file = "./trained-models/{}/{}/g_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim ,args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
		d_file = "./trained-models/{}/{}/d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_type, args.dset_name, args.encoder_dim, args.decoder_dim, args.embedding_dim, args.attention_dim, args.domain_type, args.domain_init_type, args.param_domain, args.best_k, args.weight_sim)
	if not 'generative' in args.model_type:
		model.load_state_dict(torch.load(model_file+".pt"))
		if 'gat' in args.model_type: print("W Size: {}".format(model.spatial_attn.W.size()))
		test_ade, test_fde = evaluate_model(model, testloader)
	else:
		generator.load_state_dict(torch.load(g_file+".pt"))
		discriminator.load_state_dict(torch.load(d_file+".pt"))
		test_ade, test_fde = check_accuracy(testloader, generator, discriminator, plot_traj=False, num_traj=args.num_traj)
	print(f"ADE: {test_ade:.3f}\nFDE: {test_fde:.3f}")

if __name__ == '__main__':
    main()

