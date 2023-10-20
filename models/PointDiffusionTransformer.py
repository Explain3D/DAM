#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:21:40 2023

@author: tan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointAttentionHead(nn.Module):
    def __init__(self, point_dim, latent_dim):
        super(PointAttentionHead, self).__init__()
        self.q_conv = nn.Conv1d(point_dim, latent_dim, 1, bias=False)
        self.k_conv = nn.Conv1d(point_dim, latent_dim, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(point_dim, latent_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qk, v):
        qk = qk.permute(0,2,1)
        v = v.permute(0,2,1)
        
        qk_q = self.q_conv(qk)
        qk_k = self.k_conv(qk)
        v_v = self.v_conv(v)
        qk = torch.bmm(qk_q, qk_k.permute(0,2,1)) / math.sqrt(qk_q.shape[1])
        qk = self.softmax(qk)
        qkv = torch.bmm(qk, v_v)
        return qkv


class PointDiffusionEncoder(nn.Module):
    def __init__(self, point_dim, head_dims):
        super(PointDiffusionEncoder, self).__init__()
        self.n_Head = len(head_dims)
        self.Heads = nn.ModuleList()
        for i in range(len(head_dims)):
            self.Heads.append(PointAttentionHead(point_dim=point_dim, latent_dim=head_dims[i]))
    
    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,1,self.n_Head)
        headers = []
        for i, head in enumerate(self.Heads):
            tmp_head = head(x[:,:,:,i], x[:,:,:,i])
            headers.append(tmp_head)
        headers = torch.cat(headers,dim=1)
        return headers
    
    
    
class PointDiffusionDecoder(nn.Module):
    def __init__(self, point_dim, head_dims):
        super(PointDiffusionDecoder, self).__init__()
        self.n_Head = len(head_dims)
        self.Heads = nn.ModuleList()
        for i in range(len(head_dims)):
            self.Heads.append(PointAttentionHead(point_dim=point_dim, latent_dim=head_dims[i]))
    
    def forward(self, encoded_x, x):
        encoded_x = encoded_x.permute(0,2,1)
        encoded_x = encoded_x.unsqueeze(-1).repeat(1,1,1,self.n_Head)
        x = x.unsqueeze(-1).repeat(1,1,1,self.n_Head)
        headers = []
        for i, head in enumerate(self.Heads):
            tmp_head = head(encoded_x[:,:,:,i], x[:,:,:,i])    #qk: encoded_x, v: x
            headers.append(tmp_head)
        headers = torch.cat(headers,dim=1)
        return headers
    
    
    
class PointDiffusionTransformer(nn.Module):
    def __init__(self, num_class=40):
        super(PointDiffusionTransformer, self).__init__()
        
        self.num_class = num_class
        self.encoder = PointDiffusionEncoder(3 + 259 + self.num_class, [64, 128, 256])
        self.reshape_head_1 = nn.Conv1d(448, 3 + 259 + self.num_class, 1)
        self.layer_norm_1 = nn.LayerNorm(3 + 259 + self.num_class)
        
        self.Cv1 = nn.Conv1d(3 + 259 + self.num_class,  512, 1)
        self.act_1 = F.leaky_relu
        self.Cv2 = nn.Conv1d(512, 3 + 259 + self.num_class, 1)

        
        self.decoder = PointDiffusionDecoder(3 + 259 + self.num_class, [64, 128, 256])

        self.reshape_head_2 = nn.Conv1d(448, 3 + 259 + self.num_class, 1)
        self.layer_norm_2 = nn.LayerNorm(3 + 259 + self.num_class)

        self.Cv3 = nn.Conv1d(3 + 259 + self.num_class, 512, 1)
        self.act_2 = F.leaky_relu
        self.Cv4 = nn.Conv1d(512, 3 + 259 + self.num_class, 1)
        self.act_3 = F.leaky_relu
        
        self.layer_norm_2 = nn.LayerNorm(3 + 259 + self.num_class)
        self.layer_norm_3 = nn.LayerNorm(3 + 259 + self.num_class)
        
        self.Cv5 = nn.Conv1d(3 + 259 + self.num_class, 128, 1)
        self.act_4 = F.leaky_relu
        
        self.Cv6 = nn.Conv1d(128, 3, 1)
        
        
    
    def forward(self, x, beta, context, label):
        num_point = x.shape[1]
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)
        label_emb = F.one_hot(label.long(), num_classes=self.num_class).type(torch.float)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        label_emb = label_emb.repeat(1,num_point,1)
        ctx_emb = ctx_emb.repeat(1,num_point,1)
        x = torch.cat([x, label_emb, ctx_emb], dim=-1)

    
        #Encode
        encode = self.encoder(x)
        encode = self.reshape_head_1(encode).permute(0,2,1)    
        x_encode = x + encode
        x_encode = self.layer_norm_1(x_encode)
        
        x_encode = x_encode.permute(0,2,1)
        x_encode = self.act_1(self.Cv1(x_encode))
        x_encode = self.act_2(self.Cv2(x_encode))

        decode = self.decoder(x_encode, x)
        decode = self.reshape_head_2(decode).permute(0,2,1)
        x_decode = x + decode
        x_decode = self.layer_norm_2(x_decode)
        
        x_decode = x_decode.permute(0,2,1)
        x_decode_n = self.act_2(self.Cv3(x_decode))
        x_decode_n = self.act_3(self.Cv4(x_decode_n))
        x_final = x_decode_n + x_decode

        x_final = x_final.permute(0,2,1)
        x_final = self.layer_norm_3(x_final)
        
        x_final = x_final.permute(0,2,1)
        x_final = self.act_4(self.Cv5(x_final))
        x_final = self.Cv6(x_final)
        x_final =  x_final.permute(0,2,1)
        return x_final
        
    
    
if __name__ == "__main__":
    x = torch.randn([2,1024,3])
    beta = torch.randn([2,1])
    context = torch.randn([2,256])
    label = torch.tensor([[0],[0]])
    
    
    net = PointDiffusionTransformer()
    res = net(x, beta, context, label)
    

