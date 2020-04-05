# -*- coding: utf-8 -*-
"""
test.py

@author: Vrushali Ghodnadikar
"""


import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utilities import *
import os
from model import net_G, net_D


import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import parameters
import visdom


def tester(args):
    print ("Evaluating Thopte and Prabhdeep's Model...")


    image_saved_path = parameters.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom == True:
        vis = visdom.Visdom()

    save_file_path = parameters.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path+'/'+'G.pth'
    pretrained_file_path_D = save_file_path+'/'+'D.pth'
    
    print (pretrained_file_path_G)

    D = net_D(args)
    G = net_G(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    
    print ('visualizing sid')
    
    G.to(parameters.device)
    D.to(parameters.device)
    G.eval()
    D.eval()

    N = 8

    for i in range(N):
        z = generateZ(args, 1)
        
        fake = G(z)
        samples = fake.unsqueeze(dim=0).detach().numpy()

        y_prob = D(fake)
        y_real = torch.ones_like(y_prob)
        if args.use_visdom == False:
            SavePloat_Voxels(samples, image_saved_path, 'tester__'+str(i))
        else:
            plotVoxelVisdom(samples[0,:], vis, "tester_"+str(i))