#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import time
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import struct
from write_ply import write_pointcloud
from torch.autograd import Variable
import os
import importlib
import torch.optim as optim
import sys
import random
from models.vae_flow import *
# =============================================================================
# from models.PWN_flow import *
# =============================================================================
from models.common import *
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_gen/Label_PDT_DS2502023_05_25__17_34_55/ckpt_0.000000_1020000.pt')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
# Datasets and loaders
parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ShapeNet'])
parser.add_argument('--batch_size', type=int, default=20)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=1024)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--normal', default=False)

args = parser.parse_args()

if args.device == 'cpu':
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
else:
    ckpt = torch.load(args.ckpt)

shape_names = []
SHAPE_NAME_FILE = 'data/shape_names.txt'
with open(SHAPE_NAME_FILE, "r") as f:
    for tmp in f.readlines():
        tmp = tmp.strip('\n')
        shape_names.append(tmp)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))


batch_size = args.batch_size
output_folder = "AM_output" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
if args.dataset == 'ModelNet40':
    num_class = 40
elif args.dataset == 'ShapeNet':
    num_class = 55
lr = 5e-6
repeat_time = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = PDT(ckpt['args']).to(args.device)

generator.load_state_dict(ckpt['state_dict'])


print("Generator: ", generator, "\n\n")

MODEL = importlib.import_module('pointnet_cls')

classifier = MODEL.get_model(num_class,normal_channel=False)  #PN and PN2
classifier = classifier.eval()


if args.dataset == 'ModelNet40':
    cls_path = 'log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth'
elif args.dataset == 'ShapeNet':
    cls_path = 'log_classifier/classification_shapenet/pointnet_cls_shapenet/checkpoints/best_model.pth'


checkpoint_cls = torch.load(cls_path, map_location=torch.device(args.device))
classifier.load_state_dict(checkpoint_cls['model_state_dict'])
print("Loading model: ", cls_path)
print("Classifier: ",  classifier, "\n\n")
for param in classifier.parameters():
    param.requires_grad = False
    

MODEL_guide = importlib.import_module('pointnet_cls_t')
classifier_noised_t = MODEL_guide.get_model(num_class,normal_channel=False, channel=11) #for PN and PN2
classifier_noised_t = classifier_noised_t.eval()

if args.dataset == 'ModelNet40':
    noised_path = 'log_classifier/classification/pointnet_cls_msg/checkpoints/best_model_noised_t.pth'
elif args.dataset == 'ShapeNet':
    noised_path = 'log_classifier/classification_shapenet/pointnet_cls_shapenet/checkpoints/best_model_noised_t.pth'


checkpoint_noised_t = torch.load(noised_path, map_location=torch.device(args.device))
classifier_noised_t.load_state_dict(checkpoint_noised_t['model_state_dict'])
print("Loading model: ", noised_path)
print("Classifier: ",  classifier_noised_t, "\n\n")


for param in classifier_noised_t.parameters():
    param.requires_grad = False


for param_g in generator.parameters():
    param_g.requires_grad = False
    
for param_c in classifier.parameters():
    param_c.requires_grad = False


generator = nn.DataParallel(generator)
classifier = nn.DataParallel(classifier)
classifier_noised_t = nn.DataParallel(classifier_noised_t)

generator = generator.to(device)
classifier = classifier.to(device)
classifier_noised_t = classifier_noised_t.to(device)

for tar_label in range(num_class):
    tar_cls = tar_label
    if args.dataset == 'ModelNet40':
        print("Generating " + str(shape_names[tar_label]) + " AM instances...")
    elif args.dataset == 'ShapeNet': 
        print("Generating " + str(tar_label) + " AM instances...")
    label = torch.zeros([batch_size,1])
    label = label + tar_cls
    label = label.long().to(args.device)
    
    x = torch.randn([batch_size,1024,3]).to(args.device)
    z_mu, z_sigma = generator.module.encoding(x)
    z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  #(B, 256)
    best_gen = generator.module.AM_sample(z, label, args.sample_num_points, classifier=classifier, cls_noise_guide=classifier_noised_t, flexibility=ckpt['args'].flexibility)
    
    AM_ins = best_gen
    
    actv_aft,bf_stfx_aft,_ = classifier(AM_ins.permute(0,2,1))
            
    if use_GPU == False:
        AM_ins = AM_ins.squeeze().detach().numpy()
    else:
        AM_ins = AM_ins.squeeze().detach().cpu().numpy()
    
    for ins in range(batch_size):
        if args.dataset == 'ModelNet40':
            np.save(output_folder + '/PDT/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(ins) + '.npy',AM_ins[ins])
            write_pointcloud(output_folder + '/PDT/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(ins) + '.ply',AM_ins[ins])
        elif args.dataset == 'ShapeNet': 
            np.save(output_folder + '/PDT_ShapeNet/' + 'AM_' + str(tar_label) + '_' + str(ins) + '.npy',AM_ins[ins])
            write_pointcloud(output_folder + '/PDT_ShapeNet/' + 'AM_' + str(tar_label) + '_' + str(ins) + '.ply',AM_ins[ins])



        

