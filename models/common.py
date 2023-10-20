import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn as nn

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)    #to: transform torch.randn(std.size()).dtype to mean.dtype
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)
        
        self.l_layer = Linear(dim_in, dim_out)
        
        self.label_emb_w1 = nn.Linear(40, 128)
        self.ELU_w = nn.ELU()
        self.label_emb_w2 = nn.Linear(128, dim_out)
        
        self.label_emb_b1 = nn.Linear(40, 128, bias=False)
        
        self.ELU_b = nn.ELU()
        self.label_emb_b2 = nn.Linear(128, dim_out, bias=False)
        
        self.reshape = nn.Linear(2 * dim_out, dim_out)

    def forward(self, ctx, x, label):
        
        
        l_w = self.label_emb_w1(label)
        l_w = self.ELU_w(l_w)
        l_w = self.label_emb_w2(l_w)
        l_w = torch.sigmoid(l_w)
        
        l_b = self.label_emb_b2(self.ELU_b(self.label_emb_b1(label)))

        l_x = self.l_layer(x) 
        l_x = l_x * l_w
        l_x = l_x + l_b
        
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        
        syn = torch.cat([ret, l_x], dim=-1)

        new_ret = self.reshape(syn)

        return new_ret




def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
