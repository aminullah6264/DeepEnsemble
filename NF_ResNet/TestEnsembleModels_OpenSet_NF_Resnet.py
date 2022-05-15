from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import sys
import pdb
import os
from argparse import ArgumentParser
from torch.utils.data import Subset
from tqdm import tqdm
from nf_resnet import NF_ResNet18
from utils import classification_loss, _classification_vote
from sklearn.metrics import roc_auc_score
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# import torchsummary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_expected_calibration_error(y_pred, y_true, num_bins=15):

    prob_y, pred_y = torch.max(y_pred, axis=-1)
    correct = (pred_y == y_true).type(torch.float)
    bins = torch.linspace(start=0, end=1.0, steps=num_bins).cuda()
    bins = torch.bucketize(prob_y, boundaries=bins, right=False)

    num = 0
    for b in range(num_bins):
        mask = bins == b
        if torch.any(mask):
            num += torch.abs(torch.sum(correct[mask] - prob_y[mask]))

    return num / y_pred.shape[0]


def auc_score(inliers, outliers):
    """Computes the AUROC score w.r.t network outputs on two distinct datasets.
    Typically, one dataset is the main training/testing set, while the
    second dataset represents a set of unseen outliers.

    Args: 
        inliers (torch.tensor): set of predictions on inlier data
        outliers (torch.tensor): set of predictions on outlier data

    Returns:
        AUROC score (float)
    """
    inliers = inliers.detach().cpu().numpy()
    outliers = outliers.detach().cpu().numpy()
    y_true = np.array([0] * len(inliers) + [1] * len(outliers))
    y_score = np.concatenate([inliers, outliers])
    try:
        auc_score = roc_auc_score(y_true, y_score)
    except NameError:
        print('roc_auc_score function not defined')
        auc_score = 0.5
    return auc_score


def predict_dataset(Nets, test_loader):
    output_list = []
    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        

        outputs = []
        for net in Nets:
            outputs.append(net(data))
        outputs = torch.stack(outputs).permute(1,0,2)

        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(1)
        outputs = outputs.transpose(0, 1)
        output_list.append(outputs)

    outputs = torch.cat(output_list, dim=1)  # N,B,D

    return outputs


def eval_uncertainty(model, test_loader, outlier_loader, num_particles=None):
    """Function to evaluate the epistemic uncertainty of a sampled ensemble.
    This method computes the following metrics:

    * AUROC (AUC): AUC is computed with respect to the entropy in the
      averaged softmax probabilities, as well as the sum of the
      variance of the softmax probabilities over the ensemble.

    Args:
        num_particles (int): number of sampled particles.
            If None, then self.num_particles is used.
    """

    with torch.no_grad():
        outputs = predict_dataset(model, test_loader)
        outputs_outlier = predict_dataset(model, outlier_loader)
        probs = F.softmax(outputs, -1)
        probs_outlier = F.softmax(outputs_outlier, -1)

        mean_probs = probs.mean(0)
        mean_probs_outlier = probs_outlier.mean(0)


        entropy = torch.distributions.Categorical(mean_probs).entropy()
        entropy_outlier = torch.distributions.Categorical(
            mean_probs_outlier).entropy()

        variance = F.softmax(outputs, -1).var(0).sum(-1)
        variance_outlier = F.softmax(outputs_outlier, -1).var(0).sum(-1)

        auroc_entropy = auc_score(entropy, entropy_outlier)
        auroc_variance = auc_score(variance, variance_outlier)
    print("AUROC score (entropy): {}".format(auroc_entropy))
    print("AUROC score (variance): {}".format(auroc_variance))

    return auroc_entropy, auroc_variance


def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices

def learning_rate_scheduler(optimizer_list,initial_lr,current_epoch,gamma=0.2,interval=60):
    if current_epoch<161:
        if not current_epoch==160:
            new_lr=initial_lr*(gamma**(current_epoch//interval))

        else:
            new_lr = initial_lr * (gamma ** 3)
            pass
        for optimizer in optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr


def main():
   
  

  
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
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

    Nets = []
    for i in range(10):
        Nets.append(NF_ResNet18().to(DEVICE))
    i = 1
    for net in Nets:
        pre_trained_path = '../NF_ResNet/Checkpoints3/SingleModel_' + str(i)+ '/BestModel_OpenSet_NF_ResNet_Cifar6_.pt'   ##Your saved model path here
        state_dict = torch.load(pre_trained_path)
        net.load_state_dict(state_dict)
        net.eval()
        i+=1
        print(f'model {pre_trained_path} loaded')

    correct = 0
    preds_tensor = []
    targets_tensor = []
    for batch_idx, (data, target) in tqdm(enumerate(inlier_test_loader), total=len(inlier_test_loader), smoothing=0.9):        
        data, target = data.to(DEVICE), target.to(DEVICE)               
        with torch.no_grad():
            outputs = []
            for net in Nets:
                outputs.append(net(data))
            outputs = torch.stack(outputs).permute(1,0,2)            
            _, acc = _classification_vote(outputs, target)
            correct += acc.item()
            probs = F.softmax(outputs, dim=-1)
            preds_tensor.append(probs)
            targets_tensor.append(target)
 
    preds_tensor = torch.cat(preds_tensor, dim=0).mean(1)
    targets_tensor = torch.cat(targets_tensor, dim=0)
    ece = get_expected_calibration_error(preds_tensor, targets_tensor)

    auc_entropy, auc_variance = eval_uncertainty(Nets, inlier_test_loader, outlier_test_loader)
    print('Deep Ensemble Inliers Test Accuracy =  {:.2f}% ECE = {:.9f} AUC(Entropy) ={:.6f}  AUC(Variance) ={:.6f} \n'.format(
        100. * correct / len(inlier_test_loader.dataset), ece, auc_entropy, auc_variance))



if __name__ == '__main__':
    main()
