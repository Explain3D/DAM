import numpy as np
import os
import importlib
import torch
import sys
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import random

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def FID(generated_data, real_data, model, use_GPU):
    
    model = model.to(device)
    
    model = model.eval()

    generated_data = torch.from_numpy(generated_data).unsqueeze(0).permute(0,2,1).to(device)
    
    real_data = torch.from_numpy(real_data).unsqueeze(0).permute(0,2,1).to(device)


    with torch.no_grad():
        pred_g,_,_ = model(generated_data)
        
    if use_GPU == True:
        gen_pred = torch.argmax(pred_g, dim=1).detach().cpu().numpy()[0]
    else:
        gen_pred = torch.argmax(pred_g, dim=1).detach().numpy()[0]
    
    print("Predict generated class: ", SHAPE_NAMES[gen_pred])
    
    if use_GPU == True:
        feature_compare1_g = model.feature_compare1.detach().cpu().numpy()
        feature_compare2_g = model.feature_compare2.detach().cpu().numpy()
    else:
        feature_compare1_g = model.feature_compare1.detach().numpy()
        feature_compare2_g = model.feature_compare2.detach().numpy()
        
    mu1_g = np.mean(feature_compare1_g, axis=1)
    mu2_g = np.mean(feature_compare2_g, axis=1)
    sigma1_g = np.cov(feature_compare1_g, rowvar=False)
    sigma2_g = np.cov(feature_compare2_g, rowvar=False)
    
    with torch.no_grad():
        pred_r,_,_ = model(real_data)
        
    if use_GPU == True:
        real_pred = torch.argmax(pred_r, dim=1).detach().cpu().numpy()[0]
    else:
        real_pred = torch.argmax(pred_r, dim=1).detach().numpy()[0]
        
    print("Predict real class: ", SHAPE_NAMES[real_pred])
    
    if use_GPU == True:
        feature_compare1_r = model.feature_compare1.detach().cpu().numpy()
        feature_compare2_r = model.feature_compare2.detach().cpu().numpy()
    else:
        feature_compare1_r = model.feature_compare1.detach().numpy()
        feature_compare2_r = model.feature_compare2.detach().numpy()
    
    mu1_r = np.mean(feature_compare1_r, axis=1)
    mu2_r = np.mean(feature_compare2_r, axis=1)
    sigma1_r = np.cov(feature_compare1_r, rowvar=False)
    sigma2_r = np.cov(feature_compare2_r, rowvar=False)
    
    FID_dis_fc1 = calculate_frechet_distance(mu1_g, sigma1_g, mu1_r, sigma1_r)
    FID_dis_fc3 = calculate_frechet_distance(mu2_g, sigma2_g, mu2_r, sigma2_r)
    
    FID_dis = (FID_dis_fc1 + FID_dis_fc3)/2

    return FID_dis, gen_pred, real_pred

        
    
    



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


dataset = 'ModelNet40'


if dataset == 'ModelNet40':
    num_class = 40
    
elif dataset == 'ShapeNet':
    num_class = 55
    
    
n_points = 1024
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join('data/shape_names.txt'))] 

if torch.cuda.is_available() == True:
    use_GPU = True
else:
    use_GPU = False

if(use_GPU):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#Load generated data
datapath = "AM_output/PWN_label/"
data_files = os.listdir(datapath)
generated_samples = []
gt_label_repo = []
for f in data_files:
    if f[-4:] == '.npy':
        subscript_prev_idx = f.find('_')
        subscript_later_idx = f.rfind('_')
        class_name = f[subscript_prev_idx + 1: subscript_later_idx]
# =============================================================================
#         subscript_later_idx = f.find('_')
#         class_name = f[subscript_later_idx + 5: -5]
# =============================================================================
        if dataset == 'ModelNet40':
            class_label = SHAPE_NAMES.index(class_name)
        elif dataset == 'ShapeNet':
            class_label = [int(class_name)]
            
        gt_label_repo.append(class_label)
        cur_data = np.load(datapath + f)
        generated_samples.append(cur_data)
    
generated_samples = np.asarray(generated_samples)
generated_samples = generated_samples[:]      
                  #[num_ins,1024,3]

real_data_path = 'data/modelnet40_normal_resampled/'

#Load classifier
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))
model_name = 'pointnet_FID'
MODEL = importlib.import_module(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#generator = model.PointCloudAE_4l_de(point_size,latent_size)
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()
if dataset == 'ModelNet40':
    checkpoint = torch.load('log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
elif dataset == 'ShapeNet':
    checkpoint = torch.load('log_classifier/classification_shapenet/pointnet_cls_shapenet/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])


def cal_FID(generated_data, generated_labels, real_data_path, model, use_GPU, k=5):
    model = model.to(device)
    FID_total = 0
    AM_suc_num = 0
    for ins in range(generated_data.shape[0]):
        print("Processing number ", ins, " AM instance ...")
        cur_data = generated_data[ins]
        cur_label = generated_labels[ins]
        #Check whether gt_label == pred_label
        cur_data_tmp = torch.from_numpy(cur_data).unsqueeze(0).permute(0,2,1).to(device)
        pred_check, _, _ = model(cur_data_tmp)
        if use_GPU == False:
            pred_label = torch.argmax(pred_check,axis=1)[0].detach().numpy()
        else:
            pred_label = torch.argmax(pred_check,axis=1)[0].detach().cpu().numpy()
        if pred_label != cur_label:
            print("GT is ", cur_label, ', but predicted as ', pred_label)
            print("Label check fails, skip current instance...")
            continue
        else:
            AM_suc_num += 1
            #Read real files
            class_path = real_data_path + str(SHAPE_NAMES[cur_label])+ '/'
            class_file = os.listdir(class_path)
            selected_real_data = random.sample(class_file,k)
            
            cur_gen_FID = 0
            valid = 0
            for i in range(k):
                print("Processing ", i + 1, "of ", k)
                cur_real_data = np.loadtxt(class_path + selected_real_data[i], delimiter=',').astype(np.float32)
                cur_sampled_real = farthest_point_sample(cur_real_data, n_points)
                cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
                cur_sampled_real = cur_sampled_real[:, 0:3]
                cur_FID, gen_pred, real_pred = FID(cur_data, cur_sampled_real, model, use_GPU)
                if gen_pred == real_pred:
                    cur_gen_FID += cur_FID
                    valid += 1
                else:
                    print("Wrong prediction, current FID discarded...")
            if valid == 0:    #AM generaton failed
                print("No valid AM instances available...")
                continue
            cur_gen_FID /= valid
            print("FID of current instance: ", cur_gen_FID)
            FID_total += cur_gen_FID
    FID_total /= AM_suc_num
    AM_suc_rate = AM_suc_num / generated_data.shape[0]
    return FID_total, AM_suc_rate
    
            
total_FID, AM_suc_rate = cal_FID(generated_samples, gt_label_repo, real_data_path, classifier, use_GPU=True, k=5)
print(datapath)
print("AM success rate :", AM_suc_rate)
print("Total FID: ", total_FID)
