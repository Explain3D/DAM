#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:59:00 2023

@author: tan
"""
import os
import numpy as np
import torch
import sys
import importlib
from torchmetrics.functional import spearman_corrcoef

def SC_eval(exp):
    cur_sc = []
    for t in range(1, exp.shape[0]):
        exp_t = exp[t,:]
        exp_t_minus_1 = exp[t-1,:]
        exp_t = torch.Tensor(exp_t)
        exp_t_minus_1 = torch.Tensor(exp_t_minus_1)
        sc = spearman_corrcoef(exp_t, exp_t_minus_1)
        cur_sc.append(sc)
    mean_sc = np.mean(cur_sc)
    return mean_sc


def coh_eval(exp):
    total_diff = 0
    
    for p in range(exp.shape[1]):
        cur_p = exp[:,p]
        cur_diff = 0
        
        exp_var = np.var(cur_p)
        
        for step in range(1, exp.shape[0]):
    
            cur_diff += (cur_p[step]- cur_p[step-1])
    
    
        total_diff += cur_diff
    
    print(exp_var)
    mean_diff = total_diff / ((exp.shape[0]-1) * exp.shape[1])
    print(mean_diff)
    return exp_var, mean_diff

def smooth_loss(exp):
    total_dif = 0
    for t in range(0, exp.shape[0]):
        if t == 0:
            sw = exp[:2, :]
        elif t == exp.shape[0]-1:
            sw = exp[exp.shape[0]-2:exp.shape[0],:]
        else:
            sw = exp[t-1:t+2,:]
        mv_avg = np.mean(sw,axis=0)
        smooth_dif = np.abs(mv_avg - exp[t])
        sum_smooth_dif = np.sum(smooth_dif)
        total_dif += sum_smooth_dif
    return total_dif


EXP_path = 'visu_IG'
num_class = 40
dataset = 'ModelNet40'


for cur_cls in range(num_class):
    print('Processing class ', cur_cls)
    XT = np.load(EXP_path + '/XT_[%s].npy' % cur_cls)
    IG = np.load(EXP_path + '/IG_[%s].npy' % cur_cls)
    IGD = np.load(EXP_path + '/IGD_[%s].npy' % cur_cls)
    RDM = np.random.normal(0, 1, XT.shape)
    eval_series = [RDM, IG, IGD]


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

var_RDM = []
var_IG = []
var_IGD = []

total_dif_RDM = []
total_dif_IG = []
total_dif_IGD = []

smooth_loss_RDM = []
smooth_loss_IG = []
smooth_loss_IGD = []

SC_loss_RDM = []
SC_loss_IG = []
SC_loss_IGD = []

for cur_cls in range(num_class):
    print('Processing class ', cur_cls)
    XT = np.load(EXP_path + '/XT_[%s].npy' % cur_cls)
    IG = np.load(EXP_path + '/IG_[%s].npy' % cur_cls)
    IGD = np.load( EXP_path + '/IGD_[%s].npy' % cur_cls)
    RDM = np.random.normal(0, 1, XT.shape)
    eval_series = [RDM, IG, IGD]
    
    list_for_print = ['Random', 'IG', 'IGD']
    for exp_idx in range(0,3):
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

            var, mean_diff = coh_eval(exp[ins])
            
            smooth_l = smooth_loss(exp[ins])
            
            SC_loss = SC_eval(exp[ins])

            if exp_idx == 0:
                var_RDM.append(var)
                total_dif_RDM.append(mean_diff)
                print('Random Var and Diff: ', var, mean_diff)
                
                smooth_loss_RDM.append(smooth_l)
                print('Random smooth loss: ', smooth_l)
                
                SC_loss_RDM.append(SC_loss)
                print('Random Spearman Coef loss: ', SC_loss)
                
            elif exp_idx == 1:
                var_IG.append(var)
                total_dif_IG.append(mean_diff)
                print('IG Var and Diff: ', var, mean_diff)
                
                smooth_loss_IG.append(smooth_l)
                print('IG smooth loss: ', smooth_l)
                
                SC_loss_IG.append(SC_loss)
                print('IG Spearman Coef loss: ', SC_loss)
                
            elif exp_idx == 2:
                var_IGD.append(var)
                total_dif_IGD.append(mean_diff)
                print('IGD Var and Diff: ', var, mean_diff)
                
                smooth_loss_IGD.append(smooth_l)
                print('IGD smooth loss: ', smooth_l)
                
                SC_loss_IGD.append(SC_loss)
                print('IGD Spearman Coef loss: ', SC_loss)

mean_var_RDM = np.mean(var_RDM)
mean_diff_RDM = np.mean(total_dif_RDM)
mean_smooth_RDM = np.mean(smooth_loss_RDM)
mean_SC_RDM = np.mean(SC_loss_RDM)

mean_var_IG = np.mean(var_IG)
mean_diff_IG = np.mean(total_dif_IG)
mean_smooth_IG = np.mean(smooth_loss_IG)
mean_SC_IG = np.mean(SC_loss_IG)

mean_var_IGD = np.mean(var_IGD)
mean_diff_IGD = np.mean(total_dif_IGD)
mean_smooth_IGD = np.mean(smooth_loss_IGD)
mean_SC_IGD = np.mean(SC_loss_IGD)

print('Mean var, diff, smooth and Spearman loss for Random: ', mean_var_RDM, mean_diff_RDM, mean_smooth_RDM, mean_SC_RDM)
print('Mean var, diff, smooth and Spearman loss for IG: ', mean_var_IG, mean_diff_IG, mean_smooth_IG, mean_SC_IG)
print('Mean var, diff, smooth and Spearman loss for IGD: ', mean_var_IGD, mean_diff_IGD, mean_smooth_IGD, mean_SC_IGD)



