import imp
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
        if _voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
            confidence, _ = pred.max(-1)
            confidence = probs.reshape(-1).cpu()

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
        return [], correct, confidence


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
confedence = []
correct = 0
for batch_idx, (data, target) in tqdm(enumerate(inlier_test_loader), total=len(inlier_test_loader), smoothing=0.9):        
    data, target = data.to(DEVICE), target.to(DEVICE)               
    with torch.no_grad():
        output, _ = EnsembleNet(data)
        _, acc, conf_ = _classification_vote(output, target)
        correct += acc.item()
        confedence.append(conf_)
print('Inliers Test Accuracy =  {:.2f}% for Cifar10 with \n'.format(100. * correct / len(inlier_test_loader.dataset)))


confedence = torch.cat(confedence)

confedence = confedence[confedence >0.05]

plt.figure() # Push new figure on stack
# Density Plot and Histogram of all arrival delays
sns.distplot(confedence, hist=True, kde=True)


confedence = []
correct = 0
for batch_idx, (data, target) in tqdm(enumerate(outlier_test_loader), total=len(outlier_test_loader), smoothing=0.9):        
    data, target = data.to(DEVICE), target.to(DEVICE)               
    with torch.no_grad():
        output, _ = EnsembleNet(data)
        _, acc, conf_ = _classification_vote(output, target)
        correct += acc.item()
        confedence.append(conf_)




confedence = torch.cat(confedence).cpu()
confedence = confedence[confedence >0.05]
  
print('Outliers Test Accuracy =  {:.2f}% for Cifar10 with \n'.format(100. * correct / len(outlier_test_loader.dataset)))

sns.distplot(confedence, hist=True, kde=True)

plt.legend(  labels=["Inliers","Outliers"])
plt.xlabel('Confedence')

plt.savefig('DeepEnsembleoutputUncertintyPercentaile.png') # Save that figure



# index_ = [5, 20, 40, 60, 80, 95]

# perc_func = np.percentile(confedence, index_)

# sns.distplot(confedence, hist=True, kde=True)
# for i in range(len(index_)):
#     plt.axvline(perc_func[i])

# plt.savefig('DeepEnsemblePercentaile.png') # Save that figure