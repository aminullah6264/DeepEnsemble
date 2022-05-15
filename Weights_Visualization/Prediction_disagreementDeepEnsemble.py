import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'NF_ResNet')))
# sys.path.append(os.path.abspath(os.path.join('..', 'ResNet_BatchNorm')))
import seaborn as sns
from matplotlib import pyplot as plt
from nf_resnet import CifarEnsembleRes
# from models import CifarEnsembleRes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


def _classification_vote(output, target, _voting = 'soft'):
        """Ensemble the ooutputs from sampled classifiers."""
        num_particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        
        return probs.argmax(-1)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                             [0.2023, 0.1994, 0.2010])
                                    ])

dataset2 = torchvision.datasets.CIFAR10('../NF_ResNet/cifar10Data', train=False,
                transform=transform)



label_idx = [0, 1, 2, 3, 4, 5]
outlier_label_idx = [6, 7, 8, 9]
inlier_testset = Subset(dataset2, get_classes(dataset2, label_idx))
outlier_testset = Subset(dataset2, get_classes(dataset2, outlier_label_idx))


inlier_test_loader = torch.utils.data.DataLoader(inlier_testset, batch_size=512)
outlier_test_loader = torch.utils.data.DataLoader(outlier_testset, batch_size=512)


EnsembleNet = CifarEnsembleRes(num_ensembles = 10).to(DEVICE)

pre_trained_path = '../NF_ResNet/cifar10DeepEnsemble_OpenSet_NF_ResNet_auc_entropyCifar6.pt'   ##Your saved model path here
state_dict = torch.load(pre_trained_path)
EnsembleNet.load_state_dict(state_dict)
print(f'model {pre_trained_path} loaded')

EnsembleNet.eval()

num_particles = 10

predictions = []
correct = 0

for batch_idx, (data, target) in tqdm(enumerate(inlier_test_loader), total=len(inlier_test_loader), smoothing=0.9):        
    data, target = data.to(DEVICE), target.to(DEVICE)               
    with torch.no_grad():
        output, _ = EnsembleNet(data)
        probs = F.softmax(output, dim=-1)  # [B, N, D]        
        probs = probs.argmax(-1).permute(1,0)
        predictions.append(probs)


for batch_idx, (data, target) in tqdm(enumerate(outlier_test_loader), total=len(outlier_test_loader), smoothing=0.9):        
    data, target = data.to(DEVICE), target.to(DEVICE)               
    with torch.no_grad():
        output, _ = EnsembleNet(data)
        probs = F.softmax(output, dim=-1)  # [B, N, D]        
        probs = probs.argmax(-1).permute(1,0)
        predictions.append(probs)

        
predictions = torch.cat(predictions,1).cpu()

empty_arr = np.zeros(shape=(10,10))

for i in range(10):
  preds1 = predictions[i, :]
  for j in range(i, 10):
    preds2 = predictions[j, :]
    # import ipdb; ipdb.set_trace()
    # compute dissimilarity
    dissimilarity_score = 1-torch.sum(np.equal(preds1, preds2))/10000 
    
    empty_arr[i][j] = dissimilarity_score
    if i is not j:
      empty_arr[j][i] = dissimilarity_score

dissimilarity_coeff = empty_arr[::-1]



plt.figure(figsize=(10,8))
sns.heatmap(dissimilarity_coeff, cmap='RdBu_r');
plt.xticks(np.arange(0, num_particles, step=1), np.arange(0, num_particles, 1));
plt.yticks(np.arange(0, num_particles, step=1), np.arange(num_particles, 0, -1));



plt.savefig('DeepEnsembleindependent_prediction_disagreement.png')