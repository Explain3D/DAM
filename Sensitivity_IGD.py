#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:39:47 2023

@author: tan
"""

import numpy as np
import importlib
import torch
import sys
import os
from sklearn.linear_model import LinearRegression

def reverse_points(points,explain,start='positive',percentage=0.05):
    res = np.copy(points)
    if start == 'positive':
        index_to_reverse = np.argsort(explain)[::-1]

    elif start == 'negative':
        index_to_reverse = np.argsort(explain)

    else:
        print('Wrong start input!')
        return res
    to_rev_list = explain
    num_of_rev = int(len(to_rev_list)*percentage)
    for i in range(num_of_rev):
        rev_points_index = index_to_reverse[i]
        res[rev_points_index] = [0,0,0]
    res = np.delete(res,np.argwhere(res==[0,0,0]),0)
    return res

num_class = 40
ablation_step = 20  #10 or 20
dataset = 'ModelNet40'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))

if dataset == 'ModelNet40':
    cls_path = 'log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth'
elif dataset == 'ShapeNet':
    cls_path = 'log_classifier/classification_shapenet/pointnet_cls_shapenet/checkpoints/best_model.pth'

MODEL = importlib.import_module('pointnet_cls')
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()

checkpoint_cls = torch.load(cls_path, map_location=torch.device(device))
classifier.load_state_dict(checkpoint_cls['model_state_dict'])
print("Loading model: ", cls_path)
print("Classifier: ",  classifier, "\n\n")

area_RDM = []
area_IG = []
area_IGD = []

for cur_cls in range(num_class):
    print('Processing class ', cur_cls)
    XT = np.load('visu_IG/XT_[%s].npy' % cur_cls)
    IG = np.load('visu_IG/IG_[%s].npy' % cur_cls)
    IGD = np.load('visu_IG/IGD_[%s].npy' % cur_cls)
    RDM = np.random.normal(0, 1, XT.shape)
    eval_series = [RDM, IG, IGD]
    
    list_for_print = ['Random', 'IG', 'IGD']
    for exp_idx in range(3):
        print('EXP method: ', list_for_print[exp_idx])
        exp = np.sum(eval_series[exp_idx],axis=-1)
        for ins in range(XT.shape[0]):
            print('Ablating instance ', ins)
            
            print('Conduct AM Check...')
            AM_check = torch.Tensor(XT[ins][-1]).unsqueeze(0).permute(0,2,1)
            AM_pred = classifier(AM_check)[0][0]
            if device == 'cpu':
                AM_pred_label = torch.argmax(AM_pred).detach().numpy()
            else:
                AM_pred_label = torch.argmax(AM_pred).detach().cpu().numpy()
            
            if AM_pred_label != cur_cls:
                print("AM Check Failed!")
                continue
            
            print("AM Check succeeds!")
            for step in range(XT.shape[1]):
                cur_gen = XT[ins][step]
                cur_exp = exp[ins][step]
                pos_pred_list = []
                neg_pred_list = []
                
                ini_pred = classifier(torch.Tensor(cur_gen).unsqueeze(0).permute(0,2,1))[1][0][cur_cls]
                
                if device == 'cpu':
                    pos_pred_list.append(ini_pred.detach().numpy())
                    neg_pred_list.append(ini_pred.detach().numpy())
                else:
                    pos_pred_list.append(ini_pred.detach().cpu().numpy())
                    neg_pred_list.append(ini_pred.detach().cpu().numpy())
                    
                for ablt in range(ablation_step):
                    pos_ablation = reverse_points(cur_gen, cur_exp, 'positive', percentage = 0.05 * ablt)
                    pos_pred = classifier(torch.Tensor(pos_ablation).unsqueeze(0).permute(0,2,1))[1][0][cur_cls]
                    
                    neg_ablation = reverse_points(cur_gen, cur_exp, 'negative', percentage = 0.05 * ablt)
                    neg_pred = classifier(torch.Tensor(neg_ablation).unsqueeze(0).permute(0,2,1))[1][0][cur_cls]
                    
                    if device == 'cpu':
                        pos_pred_list.append(pos_pred.detach().numpy())
                        neg_pred_list.append(neg_pred.detach().numpy())
                    else:
                        pos_pred_list.append(pos_pred.detach().cpu().numpy())
                        neg_pred_list.append(neg_pred.detach().cpu().numpy())
                    
                
                pos_pred_array = np.array(pos_pred_list)
                neg_pred_array = np.array(neg_pred_list)
                cur_area = np.trapz(y=neg_pred_array - pos_pred_array)
                
                if exp_idx == 0:
                    area_RDM.append(cur_area)
                    print('Random area: ', cur_area)
                elif exp_idx == 1:
                    area_IG.append(cur_area)
                    print('IG area: ', cur_area)
                elif exp_idx == 2:
                    area_IGD.append(cur_area)
                    print('IGD area: ', cur_area)

mean_area_RDM = np.mean(area_RDM)
mean_area_IG = np.mean(area_IG)
mean_area_IGD = np.mean(area_IGD)

print('Mean W for Random: ', mean_area_RDM)
print('Mean W for IG: ', mean_area_IG)
print('Mean W for IGD: ', mean_area_IGD)
        
    




