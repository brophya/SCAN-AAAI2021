import torch
from torch.utils.data import DataLoader
from utils import *
from arguments import *
from data import *
from model import *
import glob

def evaluate_collisions(testdataset,net,test_batch_size,thresholds):
	net.eval()
	ade = float(0)
	mean_error = float(0)
	fde = float(0)
	testloader = DataLoader(testdataset,batch_size=test_batch_size,collate_fn=collate_function(),shuffle=False)
	numTest=len(testloader)
	coll_array=[]
	for threshold in thresholds:
		print("Evaluating collisions for threshold: {}".format(threshold))
		with torch.no_grad():
			num_coll = float(0)
			num_total = float(0)
			for b, batch in enumerate(testloader):
				pred, target, sequence, pedestrians = predict(batch, net)
				pred = pred.squeeze(0).permute(1, 0, 2)
				dist_matrix = get_distance_matrix(pred, 1)
				count = torch.where(dist_matrix<threshold, torch.ones_like(dist_matrix), torch.zeros_like(dist_matrix))
				count = count.sum()-pedestrians*pred.size(0)
				count = count/2 # each collision is counted twice
				count = count.item()
				if (count>0):
					num_coll+=1
				num_total += 1 
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}")
			num_coll_percent = (num_coll/num_total)*100
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}; % collisions: {num_coll_percent}%") 
		coll_array+=[num_coll_percent]
	return coll_array


if __name__ == '__main__':
	args = parse_arguments()
	model = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_type=args.domain_type, domain_parameter=args.domain_parameter, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, pretrained_scene="resnet18", device=device, noise_dim=None, noise_type=None).float().to(device)
	model_file = f"./trained-models/{args.model_type}/{args.dset_name}/{args.encoder_dim}-{args.decoder_dim}-{args.embedding_dim}-{args.attention_dim}-{args.domain_parameter}-{args.delta_bearing}-{args.delta_heading}-{args.obs_len}-{args.pred_len}"
	model.load_state_dict(torch.load(model_file+".pt"))
	testdataset = dataset(glob.glob(f"data/{args.dset_name}/test/*.txt"), args)
	thresholds = [0.1]
	#thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2] 
	evaluate_collisions(testdataset, model, 1, thresholds)
