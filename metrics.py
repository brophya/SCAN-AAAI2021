from __future__ import print_function
import os
import sys
sys.dont_write_bytecode=True
import torch
import torch.nn as nn
import math
import numpy as np
from data import *

argoverse = False

def ade(pred, targets, num_peds, eps=1e-24, argoverse=False, agent_idx=None):
    num_peds = num_peds.sum()
    dist = torch.sqrt(((pred-targets)**2).sum(dim=-1)+eps)
    dist = 15*dist
    ade = dist.sum()/(num_peds*pred.size(2))
    return ade

def fde(pred, targets, num_peds, eps=1e-24, argoverse=False, agent_idx=None):
    num_peds = num_peds.sum()
    dist = torch.sqrt(((pred[...,-1,:]-targets[...,-1,:])**2).sum(dim=-1)+eps)
    dist = 15*dist
    fde = dist.sum()/(num_peds)
    return fde
