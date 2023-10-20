"""
Author: Benny
Date: Nov 2019
"""
from utils.ModelNetDataLoader import ModelNetDataLoader
from get_shapenet import Dataset as ShapeNetDataSet
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import shutil
import math
from write_ply import write_pointcloud



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))

def dec2bin(x, max_step):
    bits = int(np.ceil(np.log2(max_step)))
    x = torch.Tensor(x).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1)#.to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def noise_schedule(num_steps, beta_1, beta_T):
    states = num_steps + 1
    t = torch.linspace(start=0, end=num_steps, steps=states, dtype=torch.float64)
    f_t = torch.cos(((t / num_steps) + beta_1) / (1 + beta_1) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0] 
    alphas_cumprod_t = alphas_cumprod[1:]
    alphas_cumprod_t_minus_1 = alphas_cumprod[:-1]
    betas = (1 - (alphas_cumprod_t / alphas_cumprod_t_minus_1))* beta_T
    return betas

def gen_diffusion_instance(points):
    t = np.random.choice(np.arange(1, args.num_steps+1), points.shape[0])
    betas = noise_schedule(args.num_steps, args.beta_1, args.beta_T)
    betas = betas.to(torch.float32)
    betas = torch.cat([torch.zeros([1]), betas], dim=0)
    alphas = 1 - betas
    log_alphas = torch.log(alphas)
    for i in range(1, log_alphas.size(0)):  # 1 to T
        log_alphas[i] += log_alphas[i - 1]
    alpha_bars = log_alphas.exp()
    alpha_bar = alpha_bars[t]
    c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)
    e_rand = torch.randn_like(points)
    points = c0 * points + c1 * e_rand
    return points, t

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    
    parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ShapeNet'])
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet2_cls_t', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=1000, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls_msg', help='experiment root')
    parser.add_argument('--num_steps', type=int, default=250)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points, target
        
        points,t = gen_diffusion_instance(points)
        
        points = points.to(args.device)
        target = target.to(args.device)
        
        t = dec2bin(t, args.num_steps)
        t = torch.Tensor(t).unsqueeze(-1).to(args.device)
        
        classifier = model.eval()
        
        if args.model == 'pointnet_cls_t' or 'pointnet2_cls_t':
            t = t.repeat(1,1,1024)
            pred, raw, _ = classifier(points,t)
            
        if args.model == 'DGCNN_t':
            t = t.squeeze(-1)
            pred = classifier(points,t)
        
        if args.model == 'PointMLP_t':
            t = t.squeeze(-1)
            pred = classifier(points,t)

        
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):

    '''HYPER PARAMETER'''

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log_classifier/')
    experiment_dir.mkdir(exist_ok=True)
    
    if args.dataset == 'ModelNet40':
        experiment_dir = experiment_dir.joinpath('classification')
    elif args.dataset == 'ShapeNet':
        experiment_dir = experiment_dir.joinpath('classification_shapenet')
        args.log_dir = 'pointnet_cls_shapenet'
        
    experiment_dir.mkdir(exist_ok=True)
    
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
        
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()


    '''DATA LOADING'''
    print('Load dataset ...')
    if args.dataset == 'ShapeNet':
        TRAIN_DATASET = ShapeNetDataSet(root='data/', dataset_name='shapenetcorev2', num_points=1024, split='train')
        TEST_DATASET = ShapeNetDataSet(root='data/', dataset_name='shapenetcorev2', num_points=1024, split='test')
    
    
    elif args.dataset == 'ModelNet40':
        TRAIN_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=args.num_points, split='train',
                                                             normal_channel=args.normal)
        TEST_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=args.num_points, split='test',
                                                            normal_channel=args.normal)
    

        
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=1)



    '''MODEL LOADING'''
    if args.dataset == 'ModelNet40':
        num_class = 40
        MODEL = importlib.import_module(args.model)
        shutil.copy('./models/classifier/%s.py' % args.model, str(experiment_dir))
        shutil.copy('./models/classifier/pointnet_util.py', str(experiment_dir))
        
        if args.model == 'pointnet2_cls_t' or 'pointnet_cls_t':
            classifier = MODEL.get_model(num_class,normal_channel=args.normal,channel=11)
            criterion = MODEL.get_loss()
    
        elif args.model == 'DGCNN_t':
            classifier = MODEL.DGCNN_cls(k=5, emb_dims=1024, dropout=0.5, output_channels=40, device=args.device)
            criterion = MODEL.cal_loss
            
        elif args.model == 'PointMLP_t':
            classifier = MODEL.pointMLP()
            criterion = MODEL.cal_loss
    
        classifier = classifier.to(args.device)
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model_noised_t.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0
            
    elif args.dataset == 'ShapeNet':
        num_class = 55
        MODEL = importlib.import_module(args.model)
        shutil.copy('./models/classifier/%s.py' % args.model, str(experiment_dir))
        shutil.copy('./models/classifier/pointnet_util.py', str(experiment_dir))
    
        classifier = MODEL.get_model(num_class,normal_channel=args.normal, channel=11)
        criterion = MODEL.get_loss()
    
        classifier = classifier.to(args.device)
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model_noised_t.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    print('Start training...')
    

    
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
            best_epoch = epoch + 1

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
        print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

        if (instance_acc >= best_instance_acc):
            print('Save model...')
            savepath = str(checkpoints_dir) + '/best_model_noised_t.pth'
            print('Saving at %s'% savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        global_epoch += 1
    
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))


        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                print('Save model...')
                savepath = str(checkpoints_dir) + '/best_model_noised_t.pth'
                print('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data
            
            
            points, t = gen_diffusion_instance(points)
            
            points = points.numpy()
            
            
            points = random_point_dropout(points)
            points[:,:, 0:3] = random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points).to(args.device)
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points, target
            optimizer.zero_grad()
            classifier = classifier.train()
            t = dec2bin(t, args.num_steps)
            t = torch.Tensor(t).unsqueeze(-1).to(args.device)
            target = target.to(args.device)
            if args.model == 'pointnet_cls':
                t = t.repeat(1,1,1024)
                pred, _, trans_feat = classifier(points, t)
                loss = criterion(pred, target.long(), trans_feat)
            elif args.model == 'DGCNN_t':
                t = t.squeeze(-1)
                pred = classifier(points, t)
                loss = criterion(pred, target.long())
            elif args.model == 'PointMLP_t':
                t = t.squeeze(-1)
                pred = classifier(points, t)
                loss = criterion(pred, target.long())
            
            

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                print('Save model...')
                savepath = str(checkpoints_dir) + '/best_model_noised_t.pth'
                print('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    print('End of train+ing...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
