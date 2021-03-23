from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import torch
import pickle
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from utils import *
from multiprocessing import Process
import time
import random
from tqdm import tqdm
from termcolor import colored

device=torch.device("cuda" if torch.cuda.is_available else "cpu")

agroverse = True

class dataset(Dataset):
	def __init__(self,filenames,args):
		"""
		Dataset for Pedestrian Intent Modeling
		"""
		super(dataset,self).__init__()
		self.files = filenames
		self.len = -1
		self.samples=[]
		self.obs_len=args.obs_len
		self.pred_len=args.pred_len
		self.shift=1
		self.threshold=0.002
		self.delim=args.delim
		self.scene_contexts={}
		pbar = tqdm(total=len(filenames), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
		for f,filename in enumerate(filenames):
			df, means, var=self.load_data(filename)
			self.get_sequences(df, filename, means, var)
			#print(f"Processing scene context for {filename}")
			self.scene_contexts[filename] = self.get_scene_context(filename) 
			pbar.set_description(f"Processing {filename} Total Samples: {self.len}")
			pbar.update(1)
		pbar.close()
		
	def __len__(self):
		return self.len
	def load_data(self,filename):
		columns = ['t','ped id','x','y']
		data=pd.read_csv(filename,header=None,delimiter=self.delim,names=columns, dtype={'t': np.float64, 'ped id': np.int32, 'x': np.float64, 'y': np.float64})
		data.columns = ['t','ped id','x','y']
		data.sort_values(['t'],inplace=True)
		data=data[['t','ped id','x','y']]
		data['x']=data['x']-data['x'].min()
		data['y']=data['y']-data['y'].min()
		means = [data['x'].mean(), data['y'].mean()]
		var = [data['x'].max(), data['y'].max()]
		return data, means, var
	def get_sequences(self,df, fname, means, var):
		j=0
		timestamps=df['t'].unique()
		while not (j+self.obs_len+self.pred_len)>len(timestamps):
			frameTimestamps=timestamps[j:j+self.obs_len+self.pred_len]
			frame=df.loc[df['t'].isin(frameTimestamps)]
			if (not np.amax(np.diff(frameTimestamps))>10):
					# use df mean, var to normalize if using scene_context in model, comment out other
					sequence, mask, pedestrians, mean, var = self.get_sequence(frame, means, var)
					#sequence, mask, pedestrians, mean, var = self.get_sequence(frame)
					if not (pedestrians.data==0).any(): 
						self.len+=1
						sample={}
						sample['observation']=sequence
						sample['mask']=mask
						sample['pedestrians']=pedestrians
						sample['mean']=mean
						sample['var']=var
						sample['fname'] = fname
						#sample['scene_context']=scene_context
						self.samples+=[sample]
			j+=self.shift	
	def get_scene_context(self, fname):
		if 'crowds_zara02' in fname or 'crowds_zara03' in fname or 'crowds_zara01' in fname: scene_fname = "static_scene_context/zara1.png"
		elif 'students' in fname or 'uni_examples' in fname:scene_fname = "static_scene_context/univ.png"
		elif 'biwi_eth' in fname:scene_fname = "static_scene_context/eth.png"
		elif 'biwi_hotel' in fname:scene_fname = "static_scene_context/hotel.png"
		#print(f"Processing scene for {fname} from {scene_fname}") 
		return preprocess_image(scene_fname)
	def get_sequence(self,frame, means=None, var=None):
		if means is None:
			frame['x'] = frame['x']-frame['x'].min()
			frame['y'] = frame['y']-frame['y'].min()
			means = [frame['x'].mean(), frame['y'].mean()]
		
		#frame['x'] = frame['x']-means[0]
		#frame['y'] = frame['y']-means[1]
		if var is None:
			#var = [max([abs(frame['x'].max()), abs(frame['x'].min())]), max([abs(frame['y'].max()), abs(frame['y'].min())])]
			var = [frame['x'].max(), frame['y'].max()]
		frame['x'] = frame['x']/var[0] 
		frame['y'] = frame['y']/var[1] 
		frame=frame.values
		frameIDs=np.unique(frame[:,0]).tolist()	
		input_frame = frame[np.isin(frame[:,0], frameIDs[:self.obs_len])]
		pedestrians = np.unique(input_frame[:,1]).tolist()
		sequence = []
		mask = []
		sequence_=[]
		non_linear_traj=[]
		for p, pedestrian in enumerate(pedestrians):
			pedestrianTraj = frame[frame[:,1]==pedestrian]
			pedestrianTrajlen=np.shape(pedestrianTraj)[0]
			if pedestrianTrajlen<(self.obs_len+self.pred_len):
				continue
			pedestrianIDs=np.unique(pedestrianTraj[:,0])
			maskPedestrian=np.ones(len(frameIDs))
			pedestrianTraj=pedestrianTraj[:,2:]
			pedestrianTraj=pedestrianTraj+(1e-02) 
			sequence+=[torch.from_numpy(pedestrianTraj[:,:2].astype('float32')).unsqueeze(0)]
			mask+=[torch.from_numpy(maskPedestrian.astype('float32')).bool().unsqueeze(0)]
		if not sequence:
			sequence = torch.zeros(len(pedestrians),len(frameIDs),2)
			mask = torch.BoolTensor(len(pedestrians),len(frameIDs))
			pedestrians = torch.tensor(0) 
		else:
			sequence = torch.stack(sequence).view(-1,len(frameIDs),2)
			mask = torch.stack(mask).view(-1, len(frameIDs))
			pedestrians = torch.tensor(sequence.size(0))
		return sequence,mask,pedestrians,means,var 
	def __getitem__(self,idx):
		sample = self.samples[idx]
		sequence, mask, pedestrians, mean, var = sample['observation'], sample['mask'], sample['pedestrians'], sample['mean'], sample['var']
		fname = sample['fname']
		scene_context = self.scene_contexts[fname] 
		ip=sequence[:,:self.obs_len,...]
		op=sequence[:,self.obs_len:,...]
		mean, var = torch.tensor(mean).float().unsqueeze(0), torch.tensor(var).float().unsqueeze(0)
		ip_mask = mask[:,:self.obs_len]
		op_mask = mask[:,self.obs_len].unsqueeze(-1).expand(ip_mask.size(0),self.pred_len)
		ip_ = revert_orig_tensor(ip, mean, var, ip_mask)
		seq_ = revert_orig_tensor(sequence, mean, var, mask)
		dist_matrix, bearing_matrix, heading_matrix =get_features(seq_, 0, eps=0) #, mean=mean, var=var)
		return {'input':ip,'output':op[...,:2],'dist_matrix':dist_matrix,
			'bearing_matrix':bearing_matrix,'heading_matrix':heading_matrix,
			'ip_mask':ip_mask,'op_mask':op_mask,'pedestrians':pedestrians, 
			'scene_context': scene_context, 
			'mean': mean, 'var': var}


def pad_sequence(sequences,f,_len,padding_value=0.0):
	dim_ = sequences[0].size(1)
	if 'matrix' in f:
		out_dims = (len(sequences),_len,dim_,_len)
	elif 'mask' in f:
		out_dims = (len(sequences),_len,dim_)
	else:
		out_dims = (len(sequences),_len,dim_,sequences[0].size(-1))
	out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
	for i, tensor in enumerate(sequences):
		length=tensor.size(0)
		if 'matrix' in f:
			out_tensor[i,:length,:,:length]=tensor
		else:
			out_tensor[i,:length,...]=tensor
	return out_tensor

class collate_function(object):
	"""
	Custom collate function to return equal sized samples to enable batched training
	"""
	def __call__(self,batch):
		"""
		Args:
			batch: batch of unequal-sized samples
		Returns:
			output_batch: batch of equal-sized samples to enable batched dataloading and training
		"""
		batch_size=len(batch)
		features = list(batch[0].keys())
		_len = max([b['pedestrians'].data for b in batch])
		output_batch = []
		for f in features:
			if ('pedestrians' in f) or ('mean' in f) or ('var' in f) or ('scene_context' in f):
				output_feature=torch.stack([b[f] for b in batch])
			else:
				output_feature = pad_sequence([b[f] for b in batch],f,_len)
			output_batch.append(output_feature)
		return tuple(output_batch)

def poly_fit(traj, traj_len, threshold):
	t = np.linspace(0, traj_len-1, traj_len)
	res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
	res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
	if res_x+res_y>threshold:
		return 1.0
	else:
		return 0.0 
