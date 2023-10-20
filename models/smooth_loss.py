#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:49:16 2023

@author: tan
"""

import torch
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist[dist<0] = dist[dist<0] - dist[dist<0]
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist_mtx, group_idx = torch.topk(sqrdists, nsample+1, dim = -1, largest=False, sorted=False)
    return dist_mtx[:,:,1:], group_idx[:,:,1:]

def get_smooth_loss(dist_mtx, tar_dist, w):
    dist_loss = smooth_sigmoid(dist_mtx, tar_dist, w)
    dist_loss = torch.mean(dist_loss)
    return dist_loss

def smooth_sigmoid(dist, tar_dist, w):
    theta = 1e-7
    return torch.abs(torch.log((dist+theta)*(1/tar_dist))) * w

def smooth_loss(points, num_neighbor, tar_dist, w):
    dist_mtx, _ = knn_point(num_neighbor, points, points)
    dist_loss = get_smooth_loss(dist_mtx, tar_dist, w)
    return dist_loss


if __name__ == "__main__":
    points = np.load('../../tmp.npy')
    points = torch.Tensor(points)
# =============================================================================
#     points = torch.zeros([5,1024,3])
# =============================================================================
    s_loss = smooth_loss(points, 3, .0025, 1)
    print(s_loss)
    
