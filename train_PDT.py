import os
import math
import argparse
import torch
# =============================================================================
# from torch.utils.data import DataLoader
# =============================================================================
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from utils.ModelNetDataLoader import ModelNetDataLoader

from get_shapenet import Dataset as ShapeNetDataSet

from utils.dataset import *
from utils.misc import *
from utils.data import *
# =============================================================================
# from models.vae_flow import *
# =============================================================================
from models.PWN_flow import *

from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from write_ply import write_pointcloud



shape_names = []
SHAPE_NAME_FILE = 'data/shape_names.txt'
with open(SHAPE_NAME_FILE, "r") as f:
    for tmp in f.readlines():
        tmp = tmp.strip('\n')
        shape_names.append(tmp)

# Arguments
parser = argparse.ArgumentParser()
# Model arguments

parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=250)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='improved')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=1024)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders


parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ShapeNet'])

parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=64) #transformer: 16  Pointwisenet: 128
parser.add_argument('--val_batch_size', type=int, default=4)
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--normal', default=False)
parser.add_argument('--num_class', default=40)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# Training
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])

parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--W_c', type=int, default=1e-2)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=5000)
parser.add_argument('--save_freq', type=float, default=30000)
parser.add_argument('--test_size', type=int, default=128)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_inspect_pointclouds', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    if args.resume == True:
        logger = get_logger('train', args.resume_path)
        ckpt_mgr = CheckpointManager(args.resume_path)
    else:
        if args.dataset == 'ModelNet40':
            log_dir = get_new_log_dir(args.log_root, prefix='PointwiseNet_label', postfix='_' + args.tag if args.tag is not None else '')
            logger = get_logger('train', log_dir)
            ckpt_mgr = CheckpointManager(log_dir)
        
        elif args.dataset == 'ShapeNet':
            log_dir = get_new_log_dir(args.log_root, prefix='PDT_Shapenet', postfix='_' + args.tag if args.tag is not None else '')
            logger = get_logger('train', log_dir)
            ckpt_mgr = CheckpointManager(log_dir)
    

else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')


if args.dataset == 'ShapeNet':
    TRAIN_DATASET = ShapeNetDataSet(root='../Pointnet_Pytorch/data/', dataset_name='shapenetcorev2', num_points=1024, split='train')
    TEST_DATASET = ShapeNetDataSet(root='../Pointnet_Pytorch/data/', dataset_name='shapenetcorev2', num_points=1024, split='test')
    
    train_iter = get_data_iterator(DataLoader(
        TRAIN_DATASET,
        batch_size=args.train_batch_size,
        num_workers=0,
    ))
    val_loader = DataLoader(TEST_DATASET, batch_size=args.val_batch_size, num_workers=0)


elif args.dataset == 'ModelNet40':
    TRAIN_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=args.num_points, split='train',
                                                         normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=args.num_points, split='test',
                                                        normal_channel=args.normal)

    train_iter = get_data_iterator(DataLoader(
        TRAIN_DATASET,
        batch_size=args.train_batch_size,
        num_workers=0,
    ))
    val_loader = DataLoader(TEST_DATASET, batch_size=args.val_batch_size, num_workers=0)

# Model
logger.info('Building model...')

model = PDT(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch[0].to(args.device)
    label = batch[1].to(torch.float32).to(args.device)
    # Reset grad and model state

    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, label=label, writer=None, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight
    ))


def validate_inspect(it):
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        gt = batch[0].to(args.device)
        label = batch[1].to(torch.float32).to(args.device)
        model.eval()
        z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
        recons = model.sample(z, label, args.sample_num_points, flexibility=args.flexibility) #, truncate_std=args.truncate_std)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch
            
    if args.dataset == 'ModelNet40':
        for i in range(args.num_inspect_pointclouds):
            write_pointcloud("./visu_train_process/" + str(it) + "_" + str(shape_names[int(label[i].item())]) +  "_" + str(i+1) + "_gt.ply", gt[i])
            write_pointcloud("./visu_train_process/" + str(it) + "_" + str(shape_names[int(label[i].item())]) +  "_" + str(i+1) + "_rec.ply", recons[i])
        
    elif args.dataset == 'ShapeNet':
        for i in range(args.num_inspect_pointclouds):
            write_pointcloud("./visu_train_process/" + str(it) + "_" + str(int(label[i].item())) +  "_" + str(i+1) + "_gt.ply", gt[i])
            write_pointcloud("./visu_train_process/" + str(it) + "_" + str(int(label[i].item())) +  "_" + str(i+1) + "_rec.ply", recons[i])
    
# Main loop


if args.resume == True:
    logger.info('Continue training...')
    
    files = os.listdir(args.resume_path)
    max_it = -999
    latest_f = None
    for f in files:
        if f.endswith('.pt'):
            it_idx = f.rfind('_')
            f_it = int(f[it_idx+1:-3])
            if max_it < f_it:
                max_it = f_it
                latest_f = f
    
    
    print("Resuming iter ", max_it, " ...")
    
    ckpt = torch.load(args.resume_path + '/' + latest_f)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['others']['optimizer'])
    scheduler.load_state_dict(ckpt['others']['scheduler'])
    
    it_idn = args.resume_path.rfind('_')
    resume_it = max_it
    
    
else:
    logger.info('Start training...')
    
try:
    it = 1
    if args.resume == True:
        it = resume_it
    
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        if it % args.save_freq == 0:
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)

        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
