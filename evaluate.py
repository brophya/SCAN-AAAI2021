from __future__ import print_function
import torch
import glob
from torch.utils.data import DataLoader
from utils import *
from arguments import *
from data import *
from model import *

args = parse_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu_id: torch.cuda.set_device(args.gpu_id)

print("\nProcessing Test Data")
testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
testloader = DataLoader(testdataset, batch_size=args.eval_batch_size if not args.eval_batch_size is None else(testdataset), collate_fn=collate_function(), shuffle=False)

model = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_type=args.domain_type, domain_parameter=args.domain_parameter, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, pretrained_scene="resnet18", device=device, noise_dim=None, noise_type=None).float().to(device)

model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}.pt"
model.load_state_dict(torch.load(model_file))
model.eval()

ade, fde = evaluate_model(model, testloader)
print(f"ADE: {ade:.3f}\nFDE: {fde:.3f}")
