from __future__ import print_function

import sys
sys.dont_write_bytecode=True

import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import pandas as pd 
from termcolor import colored
from utils import *

class ArgoverseData(Dataset):
	def __init__(self,filenames,args):
		super(ArgoverseData, self).__init__()
		#print(filenames)
		self.files = filenames
		self.len = 0
		self.sequences={}
		self.masks={}
		self.agentCount={}
		self.sequence_length=args.obs_len
		self.prediction_length=args.pred_len
		self.feature_size=args.ip_dim
		self.output_size=args.op_dim
		self.delta_rb=args.delta_bearing
		self.delta_heading=args.delta_heading
		self.agent_idx={}
		self.idx=0
		for f,filename in enumerate(filenames):
			print(colored(f"Processing File: {f}/{len(filenames)} {filename}", "red"))
			df=self.load_data(filename)
			self.sequences[self.idx], self.masks[self.idx], self.agentCount[self.idx], self.agent_idx[self.idx] = self.get_traj(df)
			self.idx+=1
		self.len=self.idx
		sys.stdout.write("\nminimum agents per sample: %d			\n" %(min([v for k,v in self.agentCount.items()])))
		sys.stdout.write("maximum agents per sample: %d			      \n" %(max([v for k,v in self.agentCount.items()])))
	def load_data(self, filename):
		data = pd.read_csv(filename, delimiter=",", header=0)
		#columns = ['t','ped id','x','y']
		#data.columns = columns 
		data.sort_values(['TIMESTAMP'], inplace=True)
		#data=data[['t','ped id','x','y']]
		data['X'] = data['X'] - data['X'].min()
		data['Y'] = data['Y'] - data['Y'].min()
		data['X'] = data['X']/3000
		data['Y'] = data['X']/3000
		return data 
	def agent_traj(self, agent_traj):
		#agent_traj = agent_traj.values 
		agent_traj_x = agent_traj['X']
		agent_traj_y = agent_traj['Y']
		agent_traj = np.column_stack((agent_traj_x, agent_traj_y))
		return agent_traj
	def neighbor_traj(self, agent_traj, timestamps ):
		agent_timestamps = agent_traj['TIMESTAMP'].unique().tolist()
		ix_timestamps = list(range(50))
		time_to_ix = dict(zip(timestamps.tolist(), ix_timestamps))
		ix_traj = [time_to_ix[i] for i in agent_timestamps] # should give me index corresponding to timestamp existing for neighbor
		agent_x = agent_traj['X']
		agent_y = agent_traj['Y']
		agent_x = np.interp(list(range(50)), ix_traj, agent_x)
		agent_y = np.interp(list(range(50)), ix_traj, agent_y)
		agent_traj = np.column_stack((agent_x, agent_y))
		return agent_traj 
	def get_traj(self, frame):
		sequence = []
		agent_types = frame['OBJECT_TYPE'].unique()
		agents = frame['TRACK_ID'].unique()
		timestamps = frame.loc[frame['OBJECT_TYPE']=="AGENT"]['TIMESTAMP'].unique() 
		for a, agent in enumerate(agents):
			df_agent = frame.loc[frame['TRACK_ID']==agent]
			#print("AGENT DF")
			#print(df_agent.head())
			if 'AGENT' in df_agent['OBJECT_TYPE'].unique():
				agent_idx = a
				agentTraj = self.agent_traj(df_agent)
				#print("AGENT:",agentTraj.shape)
				#print(type(agentTraj))
			else:
				agentTraj = self.neighbor_traj(df_agent, timestamps)
				#print(type(agentTraj))
				#print("OTHER:",agentTraj.shape)
			sequence+=[torch.from_numpy(agentTraj).float().unsqueeze(0)]
		sequence = torch.stack(sequence).view(-1, 50, 2)
		mask = torch.ones(len(agents), 50)
		agents = torch.tensor(len(agents))
		agent_idx = torch.tensor(agent_idx)
		return sequence, mask, agents, agent_idx
	def __getitem__(self, idx):
		if not isinstance(idx,int):
				idx=int(idx.numpy())
		sequence, mask, agents = self.sequences[idx], self.masks[idx], self.agentCount[idx]
		ip = sequence[:, :self.sequence_length, ...]
		op = sequence[:, self.sequence_length:, ...]
		ip_mask = mask[:,:self.sequence_length].bool()
		op_mask = mask[:, self.sequence_length:].bool()
		dist_matrix, bearing_matrix, heading_matrix =get_features(sequence[:,:self.sequence_length,...], 0)
		assert((sequence<1).all() and (sequence>-1).all())
		return {'input':ip,'output':op[...,:self.output_size],'dist_matrix':dist_matrix,'bearing_matrix':bearing_matrix,'heading_matrix':heading_matrix,'ip_mask':ip_mask,'op_mask':op_mask,'agents':agents,'agent_idx':self.agent_idx[idx]}
	def __len__(self):
		return self.len
