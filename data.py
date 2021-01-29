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
		if (args.dset_name=="stanford"):
			self.time_delta=12
		else:
			self.time_delta=10
		self.len = -1
		self.file_idx={}
		self.frame_idx={}
		self.timestamps={}
		self.sequences={}
		self.masks={}
		self.pedestrian_count={}
		self.delim = args.delim
		self.file_num_samples={}
		self.means={}
		self.var={}
		self.scene_contexts={}
		self.sequence_length=args.obs_len
		self.prediction_length=args.pred_len
		self.feature_size=args.ip_dim
		self.output_size=args.op_dim
		self.delta_rb=args.delta_bearing
		self.delta_heading=args.delta_heading
		#self.len=0
		if 'grandcentral' in filenames[0]:
			self.shift=40
		else:
			self.shift=1
		pbar = tqdm(total=len(filenames), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
		for f,filename in enumerate(filenames):
		#	print(colored(f"Processing File: {f}/{len(filenames)} {filename}", "red"))
			df, df_mean, df_var =self.load_data(filename)
			self.timestamps[f] = df['t'].unique()
			self.file_num_samples[f] = np.int(np.floor((len(self.timestamps[f])-self.sequence_length-self.prediction_length)/self.shift))+1
			start_idx=self.len
			self.get_sequences(df,f, filename)
			for idx in range(start_idx, self.len): self.means[idx], self.var[idx] = df_mean, df_var 
			pbar.set_description(f"Processing {filename}")
			pbar.update(1)
		pbar.close()
	def __len__(self):
		return self.len
	def load_data(self,filename):
		"""
		Loads data from csv file, sorts samples based on time and scales column values to positive and 1/10th of a kilometer for faster convergence of model
		Args:
			filename: csv file
		Returns:
			data (pandas dataframe)	
		"""
		columns = ['t','ped id','x','y']
		if self.delim=="tab":
			data=pd.read_csv(filename,header=None,delimiter="\t",names=columns, dtype={'t': np.float64, 'ped id': np.int32, 'x': np.float64, 'y': np.float64})
		elif self.delim=="space":
			data=pd.read_csv(filename,header=None,delimiter=" ",names=columns, dtype={'t': np.float64, 'ped id': np.int32, 'x': np.float64, 'y': np.float64})
		elif self.delim=="comma":
			data=pd.read_csv(filename,delimiter=",", header=0) #, dtype={'TIMESTAMP': np.float64, 'TRACK_ID': np.int32, 'X': np.float64, 'Y': np.float64})
			data.columns = ['t','ped id','x','y']
		data.sort_values(['t'],inplace=True)
		data=data[['t','ped id','x','y']]
		if 'agroverse' in filename:
			data['x'] = data['x'] - data['x'].mean()
			data['y'] = data['y'] - data['y'].mean()
			data['x'] = data['x']/3000
			data['y'] = data['y']/3000
		#	print(len(data['t'].unique()))
		else:
			data['x'] = data['x'] - data['x'].min()
			data['y'] = data['y'] - data['y'].min()
			means = [data['x'].mean(), data['y'].mean()]
			data['x'] = data['x'] - data['x'].mean()
			data['y'] = data['y'] - data['y'].mean()
			var=[max([abs(data['x'].max()), abs(data['x'].min())]), max([abs(data['y'].max()), abs(data['y'].min())])]  
			data['x'] = data['x']/var[0]
			data['y'] = data['y']/var[1]
		return data, means, var
	def get_sequences(self,df,f, fname):
		"""
		Update samples 
		Args:
			df (pandas dataframe)
			f : filename 
		"""	
		j=0
		i=0
		timestamps=self.timestamps[f]
		num_samples = self.file_num_samples[f]
		while not (j+self.sequence_length+self.prediction_length)>len(timestamps):
			frameTimestamps=timestamps[j:j+self.sequence_length+self.prediction_length]
			frame=df.loc[df['t'].isin(frameTimestamps)]
			if (not np.amax(np.diff(frameTimestamps))>self.time_delta): # and (len(pedestrians)>0):
				sequence_ , mask_ , ped_count_, frame_idx_ = self.get_sequence(frame,f)
				file_idx_ = f
				scene_contexts_ = self.get_scene_context(fname) 
				if not (ped_count_.data==0).any():
					self.len+=1
					self.sequences[self.len],self.masks[self.len],self.pedestrian_count[self.len], self.frame_idx[self.len] = sequence_ , mask_ , ped_count_, frame_idx_
					self.file_idx[self.len] = file_idx_
					self.scene_contexts[self.len] = scene_contexts_
				#	sys.stdout.write(f"{j}/{len(timestamps)}\tSample id: {self.len}\tPed Count: {self.pedestrian_count[self.len]}\r")
			j+=self.shift
	def get_traj_len(self,frame,ped):
		return len(frame[frame[:,1]==ped])
	def get_scene_context(self, fname):
		if 'zara1' in fname: scene_fname = "static_scene_context/zara1.png"
		elif 'zara2' in fname: scene_fname = "static_scene_context/zara2.png"
		elif 'univ' in fname:scene_fname = "static_scene_context/univ.png"
		elif 'eth' in fname:scene_fname = "static_scene_context/eth.png"
		elif 'hotel' in fname:scene_fname = "static_scene_context/hotel.png"
		return preprocess_image(scene_fname)
	def get_sequence(self,frame,f):
		frame=frame.values
		frameIDs=np.unique(frame[:,0]).tolist()
	#	frame[:,2] = frame[:,2] - frame[:,2].mean()
	#	frame[:,3] = frame[:,3] - frame[:,3].mean()
		frame_idx=frameIDs[0]
		input_frame = frame[np.isin(frame[:,0],frameIDs[:self.sequence_length])]
		pedestrians = np.unique(input_frame[:,1]).tolist()
		sequence = []
		mask = []
		for v, pedestrian in enumerate(pedestrians):
			pedestrianTraj = frame[frame[:,1]==pedestrian]
			pedestrianTrajlen=np.shape(pedestrianTraj)[0]
			pedestrianIDs=np.unique(pedestrianTraj[:,0])
			maskPedestrian=np.ones(len(frameIDs))
			if pedestrianTrajlen<(self.sequence_length+self.prediction_length):
				continue 
			pedestrianTraj=pedestrianTraj[:,2:]
			sequence+=[torch.from_numpy(pedestrianTraj[:,:self.feature_size].astype('float32')).unsqueeze(0)]
			mask+=[torch.from_numpy(maskPedestrian.astype('float32')).bool().unsqueeze(0)]
		if not sequence:
			sequence = torch.zeros(len(pedestrians),len(frameIDs),self.feature_size)
			mask = torch.BoolTensor(len(pedestrians),len(frameIDs))
			pedestrians = torch.tensor(0) 
		else:
			sequence = torch.stack(sequence).view(-1,len(frameIDs),self.feature_size)
			mask = torch.stack(mask).view(-1, len(frameIDs))
			pedestrians = torch.tensor(sequence.size(0))
		return sequence,mask,pedestrians, frame_idx
	def __getitem__(self,idx):
		if not isinstance(idx,int):
			idx=int(idx.numpy())
		fnum = self.file_idx[idx]
		sequence,mask,pedestrians=self.sequences[idx],self.masks[idx],self.pedestrian_count[idx]
		ip=sequence[:,:self.sequence_length,...]
		op=sequence[:,self.sequence_length:,...]
		mean, var = self.means[idx], self.var[idx]     
		mean, var = torch.tensor(mean).float().unsqueeze(0), torch.tensor(var).float().unsqueeze(0)
		frame_idx = self.frame_idx[idx]
		frame_idx = torch.tensor(frame_idx).float().unsqueeze(0)
		scene_context = self.scene_contexts[idx] 
		ip_mask = mask[:,:self.sequence_length]
		op_mask = mask[:,self.sequence_length].unsqueeze(-1).expand(ip_mask.size(0),self.prediction_length)
		#if not (self.drop_frames==0):
		#	for p in range(pedestrians):
		#		perm = torch.randperm(self.sequence_length)
		#		rand_idx = perm[:self.drop_frames]
		#		ip_mask[p, rand_idx] = torch.zeros(ip_mask[p, rand_idx].size())
		dist_matrix, bearing_matrix, heading_matrix =get_features(sequence[:,:self.sequence_length,...], 0, mean=mean, var=var)
		#assert((sequence<1).all() and (sequence>-1).all())
		return {'input':ip,'output':op[...,:self.output_size],'dist_matrix':dist_matrix,'bearing_matrix':bearing_matrix,'heading_matrix':heading_matrix,'ip_mask':ip_mask,'op_mask':op_mask,'pedestrians':pedestrians, 'scene_context': scene_context, 'mean': mean, 'var': var, 'frame_id': frame_idx} 

def fillarr(arr):
	for i in range(arr.shape[1]):
		idx=np.arange(arr.shape[0])
		idx[arr[:,i]==0] = 0
		np.maximum.accumulate(idx, axis = 0, out=idx)
		arr[:,i] = arr[idx,i]
		if (arr[:,i]==0).any():
			idx[arr[:,i]==0] = 0
			np.minimum.accumulate(idx[::-1], axis=0)[::-1]
			arr[:,i] = arr[idx,i]
	return arr

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
		_len = max([b['input'].size(0) for b in batch])
		output_batch = []
		for f in features:
			if ('pedestrians' in f) or ('agents' in f) or ('agent_idx' in f) or ('mean' in f) or ('var' in f) or ('frame_id' in f) or ('scene_context' in f):
				output_feature=torch.stack([b[f] for b in batch])
			else:
				output_feature = pad_sequence([b[f] for b in batch],f,_len)
			output_batch.append(output_feature)
		return tuple(output_batch)
	
def interpolate(arr):
	cols=arr.shape[1]
	arr=pd.DataFrame(arr,index=None)
	for c in range(cols):
		arr.iloc[:,c].replace(to_replace=0,inplace=True,method='ffill')
		arr.iloc[:,c].replace(to_replace=0,inplace=True,method='bfill')
	return arr.values
