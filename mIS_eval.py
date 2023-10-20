import numpy as np
import os
import importlib
import torch
import sys
import math


def Modifified_inception_score(data, model, labels, num_cls, use_GPU = True):
    #Get predictions
    model = model.to(device)
    final_is = []
    print(data.shape[0])
    AM_suc_num = 0
    pred_mtx = []
    total_processed = 0
    for i in range(data.shape[0]):

        total_processed += 1
        print("Processing number ", i, "ins...")
        cur_data = data[i]
        print(cur_data.shape)
        cur_label = labels[i]
        cur_data = torch.from_numpy(cur_data).unsqueeze(0).permute(0,2,1).to(device)
        preds,bfsftmx,_ = model(cur_data)
        if use_GPU == True:
            pred_cls = torch.argmax(preds,axis=1).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
        else:
            pred_cls = torch.argmax(preds,axis=1).detach().numpy()
            preds = preds.detach().numpy()
        if pred_cls != cur_label:
            print("GT is ", cur_label, ', but predicted as ', pred_cls)
            print("Label check fails, skip current instance...")
            continue
        else:
            AM_suc_num += 1
            pred_mtx.append(np.exp(preds))
        
    
            
    splits = num_cls		# the number of splits to average the score over
    scores = []
    pred_mtx = np.array(pred_mtx)
    pred_mtx = np.squeeze(pred_mtx)
    argmax = np.argmax(pred_mtx,axis=1)
    # Calculating the inception score
    for i in range(splits):
        part = pred_mtx[argmax==i]
        if part.shape[0] > 0:
            logp= np.log(part)
            self = np.sum(part*logp,axis=1)
            cross = np.mean(np.dot(part,np.transpose(logp)),axis=1)
            kl = np.mean(self - cross)
            scores.append(np.exp(kl))
    print(scores)
    print("scores_mean = %.2f, scores_std = %.2f" % (np.mean(scores),
                                                     np.std(scores)))
    
    final_is = np.mean(scores)      
    AM_suc_num = AM_suc_num / total_processed
    return final_is, AM_suc_num


dataset = 'ShapeNet'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

if dataset ==  'ModelNet40':
    num_class = 40
elif dataset == 'ShapeNet':
    num_class = 55

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


datapath = "AM_output/PDT_ShapeNet/"

data_files = os.listdir(datapath)
generated_samples = []
label_repo = []
for f in data_files:
    if f[-4:] == '.npy':
        print("Reading ", f, "......")
        subscript_prev_idx = f.find('_')
        subscript_later_idx = f.rfind('_')
        class_name = f[subscript_prev_idx + 1: subscript_later_idx]
# =============================================================================
#         subscript_later_idx = f.find('_')
#         class_name = f[subscript_later_idx + 5: -5]
# =============================================================================
        print(class_name)
        if dataset == 'ModelNet40':
            class_label = SHAPE_NAMES.index(class_name)
        elif dataset == 'ShapeNet':
            class_label = [int(class_name)]
        
        label_repo.append(class_label)
        cur_data = np.load(datapath + f)
        generated_samples.append(cur_data)
    
generated_samples = np.asarray(generated_samples)
generated_samples = generated_samples[:]
print(generated_samples.shape)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/classifier'))
model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
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







m_IS, AM_acc = Modifified_inception_score(generated_samples,classifier, label_repo, num_class, use_GPU = use_GPU)
print("Modified Inception Score: ", m_IS)
print("AM success rate: ", AM_acc)
print("Path: ", datapath)
