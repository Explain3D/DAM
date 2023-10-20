import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *
# =============================================================================
# from .transformer_module import PointDiffusionTransformer
# =============================================================================
from .Point_Unet import PointUnet


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.context_dim)
# =============================================================================
#         self.diffusion = DiffusionPoint(
#             net = PointwiseNet(point_size=1024, latent_size=args.latent_size, context_dim=args.context_dim),
#             var_sched = VarianceSchedule(
#                 num_steps=args.num_steps,
#                 beta_1=args.beta_1,
#                 beta_T=args.beta_T,
#                 mode=args.sched_mode
#             )
# =============================================================================
# =============================================================================
#         self.diffusion = DiffusionPoint(
#             net = PointwiseNet(point_dim=3, context_dim=args.context_dim, residual=args.residual),
#             var_sched = VarianceSchedule(
#                 num_steps=args.num_steps,
#                 beta_1=args.beta_1,
#                 beta_T=args.beta_T,
#                 mode=args.sched_mode
#             )
#         )
# =============================================================================
# =============================================================================
#         self.diffusion = DiffusionPoint(
#             net = PointDiffusionTransformer(device=args.device, dtype=torch.float32, time_token_cond=True, width=256),
#             var_sched = VarianceSchedule(
#                 num_steps=args.num_steps,
#                 beta_1=args.beta_1,
#                 beta_T=args.beta_T,
#                 mode=args.sched_mode
#             )
#         )
# =============================================================================
        self.diffusion = DiffusionPoint(
            net = PointUnet(point_dim=3, num_neighbors=16, decimation=4, device=args.device),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, label, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, label, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, label):
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code, label)
        return loss
