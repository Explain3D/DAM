import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import math
import torch.nn as nn

from .common import *
from .smooth_loss import smooth_loss
from write_ply import write_pointcloud

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        #assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            print("Schedule: Linear")
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'improved':
            print("Schedule: Improved")
            betas = self.noise_schedule(num_steps, beta_1, beta_T)
            betas = betas.to(torch.float32)

        
        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    def noise_schedule(self, num_steps, beta_1, beta_T):
        states = num_steps + 1
        t = torch.linspace(start=0, end=num_steps, steps=states, dtype=torch.float64)
        f_t = torch.cos(((t / num_steps) + beta_1) / (1 + beta_1) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0] 
        alphas_cumprod_t = alphas_cumprod[1:]
        alphas_cumprod_t_minus_1 = alphas_cumprod[:-1]
        betas = (1 - (alphas_cumprod_t / alphas_cumprod_t_minus_1))* beta_T
        return betas



class DiffusionPoint(Module):
    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.save_process = True
        self.save_step = 50
        self.save_ply = True
        
    def cal_IG(self, x_0,x_t,model,tar_cls,step):
        total_grad = 0
        for i in range(step):
            cur_x = x_0 + i * (x_t - x_0) / step
            cur_x.requires_grad=True
            
            pred,_,_ = model(cur_x.permute(0,2,1))
            pred = pred[:, tar_cls]
            pred = torch.sum(pred)
            grad = torch.autograd.grad(torch.log(pred), cur_x, retain_graph=True)[0].data
            total_grad += grad
            
        IG = total_grad / step + (x_t - x_0)
        return IG
        
        
    def normalize(self, contribution):
        min_c = np.min(contribution)
        contribution = contribution - min_c
        max_c = np.max(contribution)
        contribution = contribution / max_c
        return contribution * 2 - 1

    def standardize(self, contribution):
        contribution = (contribution - np.mean(contribution)) / np.std(contribution)
        return contribution      
        
    def contri_to_color(self, current_data,contri):
        current_data = current_data.detach().cpu().numpy()
        contri = contri.detach().cpu().numpy()
        point_contri = np.sum(contri,axis=1)
        point_contri = self.standardize(point_contri)
        color_matrix = np.zeros([point_contri.shape[0],3])
        for i in range(point_contri.shape[0]):
            if point_contri[i] < 0:
                color_matrix[i][2] = abs(point_contri[i])
                    
            elif point_contri[i] > 0:
                color_matrix[i][0] = point_contri[i]
                
        colored_data = np.concatenate((current_data,color_matrix),axis = 1)
        return colored_data


    def dec2bin(self, x, max_step):
        bits = int(np.ceil(np.log2(max_step)))
        x = torch.Tensor(x).int()
        mask = 2 ** torch.arange(bits - 1, -1, -1)#.to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def get_loss(self, x_0, context, label, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        
        x_in = c0 * x_0 + c1 * e_rand    #x_t
        e_theta = self.net(x_in, beta=beta, context=context, label=label)
# =============================================================================
#         e_theta = self.net(x_in, beta=beta, context=context) # for PWN
# =============================================================================
        loss = F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        
        
        return loss

    def AM_sample(self, num_points, context, tar_cls, AM_lr=1e-4, point_dim=3, classifier=None, cls_noise_guide=None, flexibility=0.0, ret_traj=False, noise_guide=True):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        
        x_0 = x_T.clone().detach()
        total_grad = 0
    
        IGnpy = []
        IGDnpy = []
        
        xt_npy = []
        
    
        for t in range(self.var_sched.num_steps, 0, -1):
            
            print("Sampling step: ", t)
            
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)  
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            x_t = traj[t]
            
            cur_AM_ipt = x_t.transpose(1,2).to(context.device)
            cur_AM_ipt.requires_grad = True
            
            t_emb = self.dec2bin([t], self.var_sched.num_steps)
            t_emb = torch.Tensor(t_emb).repeat(x_t.shape[0],1).to(context.device)
            
            noise_pred, _, _ = cls_noise_guide(cur_AM_ipt,t_emb)
# =============================================================================
#             noise_pred = cls_noise_guide(cur_AM_ipt,t_emb)
# =============================================================================
            noise_pred = noise_pred[:, tar_cls]
            noise_pred = torch.sum(noise_pred)
            noise_guide_grad = torch.autograd.grad(torch.log(noise_pred), cur_AM_ipt, retain_graph=True)[0].data
            
            pred, _, _ = classifier(cur_AM_ipt)
# =============================================================================
#             pred = classifier(cur_AM_ipt)
# =============================================================================
            pred = pred[:, tar_cls]
            pred = torch.sum(pred)
            
            cls_grad = torch.autograd.grad(torch.log(pred), cur_AM_ipt, retain_graph=True)[0].data
            
            
            w_n = t / self.var_sched.num_steps
            w_c = 1 - w_n
            
            if noise_guide == True:
                AM_grad = cls_grad.transpose(1,2) * w_c + noise_guide_grad.transpose(1,2) * w_n
            else:
                AM_grad = cls_grad.transpose(1,2)


            if self.save_process == True:
                if t % self.save_step == 0:
                    total_grad += cls_grad.transpose(1,2)
                    mean_grad = total_grad / (self.var_sched.num_steps - t)
                    
                    cur_IGD = (x_t - x_0) * mean_grad
                    
                    cur_IG = self.cal_IG(x_0,x_t,classifier,tar_cls,25)
                
                
                    if t != self.var_sched.num_steps:
                        IGDnpy.append(cur_IGD.detach().cpu().numpy())
                        IGnpy.append(cur_IG.detach().cpu().numpy())
                        xt_npy.append(x_t.detach().cpu().numpy())
                        
                    if self.save_ply == True:
                        for i in range(cur_IGD.shape[0]):
                            colored_data = self.contri_to_color(x_t[i], cur_IGD[i])
                            write_pointcloud("visu_IG/IG_DF_" + str(t) + "_ins_" + str(i) + ".ply", colored_data[:,0:3], colored_data[:,3:6])
                
            x_t = x_t + (AM_grad * AM_lr)
            
            
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context, label=tar_cls)
# =============================================================================
#             e_theta = self.net(x_t, beta=beta, context=context) # for PWN
# =============================================================================
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
                

        final_exp = traj[0].clone().permute(0,2,1).to(context.device)
        final_exp.requires_grad = True
        
        pred, _, _ = classifier(final_exp)
# =============================================================================
#         pred = classifier(final_exp)
# =============================================================================
        pred = pred[:, tar_cls]
        pred = torch.sum(pred)
        cls_grad = torch.autograd.grad(torch.log(pred), final_exp, retain_graph=True)[0].data
        
        total_grad += cls_grad.transpose(1,2)
        mean_grad = total_grad / (self.var_sched.num_steps - t)
        cur_IGD = (traj[0] - x_0) * mean_grad
        
        
        if self.save_process == True:
            
            cur_IG = self.cal_IG(x_0,traj[0],classifier,tar_cls,25)
            
            
            IGDnpy.append(cur_IGD.detach().cpu().numpy())
            IGnpy.append(cur_IG.detach().cpu().numpy())
            xt_npy.append(traj[0].detach().cpu().numpy())
            
            if self.save_ply == True:
                for i in range(cur_IGD.shape[0]):
                    colored_data = self.contri_to_color(traj[0][i], cur_IGD[i])
                    write_pointcloud("visu_IG/IG_DF_" + str(0) + "_ins_" + str(i) + ".ply", colored_data[:,0:3], colored_data[:,3:6])
            
            IGDnpy = np.asarray(IGDnpy)
            IGnpy = np.asarray(IGnpy)
            xt_npy = np.asarray(xt_npy)
            
            IGDnpy = IGDnpy.transpose(1,0,2,3)
            IGnpy = IGnpy.transpose(1,0,2,3)
            xt_npy = xt_npy.transpose(1,0,2,3)
            
            str_tar_cls = tar_cls.clone().detach().cpu().numpy()
            str_tar_cls = np.unique(str_tar_cls)
            
            np.save('visu_IG/IGD_'+ str(str_tar_cls)+ '.npy', IGDnpy)
            np.save('visu_IG/IG_'+ str(str_tar_cls)+ '.npy', IGnpy)
            np.save('visu_IG/XT_'+ str(str_tar_cls)+ '.npy', xt_npy)
        
        if ret_traj:
            return traj
        else:
            return traj[0]
        
        
    def sample(self, num_points, context, label, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)  
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context, label=label)
# =============================================================================
#             e_theta = self.net(x_t, beta=beta, context=context)   #for PWN
# =============================================================================
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

