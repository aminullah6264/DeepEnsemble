

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.manifold import TSNE
import sys, os
import functools
sys.path.append(os.path.abspath(os.path.join('..', 'NF_ResNet')))
from matplotlib import pyplot as plt
import seaborn as sns
from nf_resnet import CifarEnsembleRes

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
        if _voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)

        elif _voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = []
            for i in range(pred.shape[0]):
                values, counts = torch.unique(
                    pred[i], sorted=False, return_counts=True)
                modes = (counts == counts.max()).nonzero()
                label = values[torch.randint(len(modes), (1, ))]
                vote.append(label)
            vote = torch.as_tensor(vote, device='cpu')
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        # loss = classification_loss(output, target)
        return [], correct, probs



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


inlier_test_loader = torch.utils.data.DataLoader(inlier_testset, batch_size=1024)
outlier_test_loader = torch.utils.data.DataLoader(outlier_testset, batch_size=1024)

NUM_TRAJECTORIES = 1



EnsembleNet = CifarEnsembleRes(num_ensembles = 10).to(DEVICE)
prob_ = []
for i in range(40):
  pre_trained_path = '../NF_ResNet/Checkpoints2/DeepEnsemble_OpenSet_NF_ResNet_auc_entropyCifar6_' + str(i+1) + '.pt'   ##Your saved model path here
  state_dict = torch.load(pre_trained_path)
  EnsembleNet.load_state_dict(state_dict)
  print(f'model {pre_trained_path} loaded')

  EnsembleNet.eval()
  correct = 0
  for batch_idx, (data, target) in tqdm(enumerate(inlier_test_loader), total=len(inlier_test_loader), smoothing=0.9):        
      data, target = data.to(DEVICE), target.to(DEVICE)               
      with torch.no_grad():
          output, _ = EnsembleNet(data)
          probs = F.softmax(output, dim=-1)[:,:,:].permute(1,0,2)  # [B, N, D]   --> [N, B, D]  
          prob_.append(probs)
      break


NUM_PARTICLES, NUM_EXAMPLES, NUM_CLASSES = probs.shape

predictions_for_tsne = torch.cat(prob_, dim=1).cpu()

# reshape the tensor 
reshaped_predictions_for_tsne = predictions_for_tsne.reshape(-1, NUM_EXAMPLES*NUM_CLASSES)
print('[INFO] shape of reshaped tensor: ', reshaped_predictions_for_tsne.shape)


# initialize tsne object
tsne = TSNE(n_components=2)
# compute tsne
prediction_embed = tsne.fit_transform(reshaped_predictions_for_tsne)
print('[INFO] Shape of embedded tensor: ', prediction_embed.shape)
# reshape
trajectory_embed = prediction_embed.reshape([NUM_TRAJECTORIES, -1, 2])
print('[INFO] Shape of reshaped tensor: ', trajectory_embed.shape)

# Plot
plt.figure(constrained_layout=True, figsize=(6,6))

# colors_list=['r', 'b', 'g']
from random import randint

colors_list = []
n = NUM_TRAJECTORIES

for i in range(n):
    colors_list.append('#%06X' % randint(0, 0xFFFFFF))

labels_list = ['traj_{}'.format(i) for i in range(NUM_TRAJECTORIES)]
for i in range(NUM_TRAJECTORIES):
    plt.plot(trajectory_embed[i,:,0],trajectory_embed[i,:,1],color = colors_list[i], alpha = 0.8,linestyle = "", marker = "o")
    plt.plot(trajectory_embed[i,:,0],trajectory_embed[i,:,1],color = colors_list[i], alpha = 0.3,linestyle = "-", marker = "")
    plt.plot(trajectory_embed[i,0,0],trajectory_embed[i,0,1],color = 'r', alpha = 1.0,linestyle = "", marker = "*")


plt.savefig('DeepEnsemble_loss_landscape_2d.png')







