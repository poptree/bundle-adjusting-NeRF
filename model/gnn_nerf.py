import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import nerf
import camera
from .mpnn import MPNN_3Conv

# ============================ main engine for training and evaluation ============================

class Model(nerf.Model):
    def __init__(self, opt):
        super().__init__(opt)
        
    def build_networks(self, opt):
        super().build_networks(opt)
    
        if opt.camera.noise:
            se3_noise = torch.randn(len(self.train_data), 6, device=opt.device)*opt.camera.noise
        
        self.node_feat_num = 32
        self.edge_feat_num = 1
        self.edge_index = None
        
        # self.graph.se3_refine = torch.nn.Embedding(len(self.train_data),32).to(opt.device)
        self.graph.camera_embedding = torch.nn.Embedding(len(self.train_data), 32).to(opt.device)
        self.gnn = MPNN_3Conv()
        # torch.nn.init.zeros_(self.graph.se3_refine.weight)
        torch.nn.init.normal_(self.)
        
        