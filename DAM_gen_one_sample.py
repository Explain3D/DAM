#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import time
import utils
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import struct
from utils.ModelNetDataLoader import ModelNetDataLoader
from write_ply import write_pointcloud
from torch.autograd import Variable
import os
import importlib
import torch.optim as optim
import sys
import random
from models.vae_flow import *
from write_ply import write_pointcloud
from models.common import *

    

parser = argparse.ArgumentParser()
# =============================================================================
# parser.add_argument('--ckpt', type=str, default='./logs_gen/PDT_Shapenet2023_06_29__12_05_35/ckpt_0.000000_1020000.pt')
# =============================================================================
parser.add_argument('--ckpt', type=str, default='./logs_gen/Label_PDT_DS2502023_05_25__17_34_55/ckpt_0.000000_1020000.pt')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
# Datasets and loaders
parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ShapeNet'])

# Sampling
parser.add_argument('--sample_num_points', type=int, default=1024)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--num_points', type=int, default=1024)

parser.add_argument('--normal', default=False)

args = parser.parse_args()


if args.device == 'cpu':
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
else:
    ckpt = torch.load(args.ckpt)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))

# =============================================================================
# def add_sparse_Gauss_noise(vector,noise_mean,noise_var,noise_rate=0.05, n_weight = 0):
#     v_shape = vector.size()
#     #G_noise = torch.empty(v_shape,device = device).normal_(mean=noise_mean,std=noise_var)
#     G_noise = torch.empty(v_shape).normal_(mean=noise_mean,std=noise_var)
#     noise_mask = torch.empty(v_shape).uniform_() > 0.95
#     G_noise = torch.mul(G_noise,noise_mask))
#     res = vector + torch.Tensor(1)#(n_weight * G_noise.to(device))
#     return res
# =============================================================================

shape_names = []
SHAPE_NAME_FILE = 'data/shape_names.txt'
with open(SHAPE_NAME_FILE, "r") as f:
    for tmp in f.readlines():
        tmp = tmp.strip('\n')
        shape_names.append(tmp)


batch_size = 1
output_folder = "output/" # folder path to save the results
save_results = True # save the results to output_folder
latent_size = 128 # bottleneck size of the Autoencoder model
if args.dataset == 'ModelNet40':
    num_class = 40
elif args.dataset == 'ShapeNet':
    num_class = 55
tar_label = 0
avg_initialization = True
maximize_logstfmx = False
lr = 1e-3

# =============================================================================
# from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
# =============================================================================

if args.dataset == 'ModelNet40':
    model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
elif args.dataset == 'ShapeNet':
    model_name = os.listdir('log_classifier/classification_shapenet/pointnet_cls_shapenet'+'/logs')[0].split('.')[0]

MODEL = importlib.import_module(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# ckpt['args'].device = 'cpu'
# =============================================================================
generator = PDT(ckpt['args']).to(args.device)
generator.load_state_dict(ckpt['state_dict'])
print("Generator: ", generator, "\n\n")

MODEL = importlib.import_module('pointnet_cls')
classifier = MODEL.get_model(num_class,normal_channel=False)
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
classifier_noised_t = MODEL_guide.get_model(num_class,normal_channel=False, channel=11)
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


generator = generator.to(args.device)
classifier = classifier.to(args.device)
classifier_noised_t = classifier_noised_t.to(args.device)

for param_g in generator.parameters():
    param_g.requires_grad = False

num_samples = 20
tar_cls = 33
AM_lr = 1e-4

label = torch.zeros([num_samples,1])
label = label + tar_cls
label = label.long().to(args.device)


x = torch.randn([num_samples,1024,3]).to(args.device)
z_mu, z_sigma = generator.encoding(x)
z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)


EXP = generator.AM_sample(z, label, AM_lr, args.sample_num_points, classifier=classifier, cls_noise_guide=classifier_noised_t, flexibility=ckpt['args'].flexibility)

if args.device == 'cpu':
    EXP_npy = EXP.detach().numpy()
else:
    EXP_npy = EXP.detach().cpu().numpy()

for i in range(num_samples):
    if args.dataset == 'ModelNet40':
        write_pointcloud("./visu_EXP/%s_%d.ply" % (shape_names[tar_cls], i), EXP_npy[i])
    elif args.dataset == 'ShapeNet':
        write_pointcloud("./visu_EXP/%d_%d.ply" % (tar_cls, i), EXP_npy[i])

EXP = EXP.transpose(1,2)


pred, bf_sfx, _ = classifier(EXP)
pred_cls = torch.argmax(pred, dim=1).detach().cpu().numpy()

bf_sfx = bf_sfx.detach().cpu().numpy()

for i in range(num_samples):
    print(bf_sfx[i][tar_cls], shape_names[pred_cls[i]])


