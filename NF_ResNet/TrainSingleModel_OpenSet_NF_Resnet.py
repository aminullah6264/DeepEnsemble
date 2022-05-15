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


def predict_dataset(model, test_loader):
    output_list = []
    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output, _ = model(data)

        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        output = output.transpose(0, 1)
        output_list.append(output)

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
    # epochs = 100
    # # How many Ensemble model you want to create num_ensembles = X
    # num_ensembles = 10
    # dataset = 'cifar10'  # can be 'mnist', 'f_mnist', 'cifar10'

    parser = ArgumentParser("Args for Train Ensemble eval. of ResNet-18-A")
    parser.add_argument('--uncertainty_eval', action='store_true', default=True, help='Enable uncertainty evaluation',
                        required=False)
    parser.add_argument('--dataset',type=str,default='cifar10',help='Dataset to train upon')
    parser.add_argument('--num_ensemble',type=int,default=10,help='No. of models in the ensemble')
    parser.add_argument('--epochs',type=int,default=200,help='No. of epochs to train the model')
    parser.add_argument('--SGLD',action='store_true',default=False,help='Enable SGLD optimizer')
    parser.add_argument('--debug',action='store_true',default=False,help='Enable debug mode')
    parser.add_argument('--batch_size',type=int,default=128,help='Batch size')
    parser.add_argument('--lr',type=float,default=0.1,help='initial learning rate')

    args = parser.parse_args()
    uncertainty_eval = args.uncertainty_eval if args.uncertainty_eval else False
    dataset=args.dataset
    num_ensembles=args.num_ensemble
    epochs=args.epochs
    batch_size=args.batch_size
    initial_lr=args.lr

    if dataset == 'cifar10':
        transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2023, 0.1994, 0.2010])
            ])

        if uncertainty_eval:            

            if not args.debug:
                dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True,
                                                    transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                transform ])   )

                dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                                                        transform=transform)
            else:
                dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True,
                                                    transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                transform ])   )

                dataset1,_2 =torch.utils.data.random_split(dataset1,[int(0.1*len(dataset1)),int(0.9*len(dataset1))])

                dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                                                        transform=transform)
                dataset2, _2 = torch.utils.data.random_split(dataset2,
                                                             [len(dataset1), len(dataset2)-len(dataset1)])


            label_idx = [0, 1, 2, 3, 4, 5]
            outlier_label_idx = [6, 7, 8, 9]
            trainset = Subset(dataset1, get_classes(dataset1, label_idx))
            testset = Subset(dataset2, get_classes(dataset2, label_idx))

            testset_outlier = Subset(dataset2, get_classes(dataset2, outlier_label_idx))

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=128)

            outlier_loader = torch.utils.data.DataLoader(testset_outlier, batch_size=128)
            best_auc_entropy = 0.
        else:
            dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True,
                                                    transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                transform ])   )

                
            dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                                                    transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle = True)
            test_loader = torch.utils.data.DataLoader(dataset2, batch_size=500)

        Net = NF_ResNet18().to(DEVICE)
    
    ensemble_model_num = 10

    lr_ = 0.1
    momentum_ = 0.9
    wd_ = 1e-4
    optimizer = torch.optim.SGD(Net.parameters(), lr_,
                                momentum= momentum_,
                                weight_decay= wd_)
    lossFun = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[82, 123], last_epoch= -1 )
    path = 'Checkpoints3/SingleModel_'+str(ensemble_model_num)
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

    modelName = 'Checkpoints3/SingleModel_'+str(ensemble_model_num)+'/OpenSet_NF_ResNet_Cifar6_' + str(0) + '.pt'
    print('Model Saved')
    torch.save(Net.state_dict(), modelName)
                                   

    Best_Acc = 0
    for epoch in range(1, epochs):
        train_loss = 0
        train_correct = 0

        Net.train()
        if args.SGLD:
            learning_rate_scheduler(lr_scheduler,initial_lr=initial_lr,current_epoch=epoch,gamma=0.1)
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            output = Net(data)
            loss = lossFun(output, target)
            loss.backward()
            train_loss += loss
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()


        lr_scheduler.step()
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / (len(train_loader.dataset))

        print('Epoch: {} SingleModel {} OpenSet_NF_ResNet--> {} Dataset Training Loss = {:.4f}, Train Accuracy =  {:.2f}%, \n'.format(
             epoch, ensemble_model_num, dataset, train_loss, train_acc))

        Net.eval()

        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = Net(data)
                probs = F.softmax(output, dim=-1)
                pred = probs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                

            print('Epoch: {} , Test Accuracy =  {:.2f}%  \n'.format(
                epoch, 100. * correct / len(test_loader.dataset)))

            if epoch < 41:
                modelName = 'Checkpoints3/SingleModel_'+str(ensemble_model_num)+'/OpenSet_NF_ResNet_Cifar6_' + str(epoch) + '.pt'
                print('Model Saved')
                torch.save(Net.state_dict(), modelName)
            
            if train_acc > Best_Acc:
                modelName = 'Checkpoints3/SingleModel_'+str(ensemble_model_num)+'/BestModel_OpenSet_NF_ResNet_Cifar6_.pt'
                print('Model Saved')
                torch.save(Net.state_dict(), modelName)
                Best_Acc = train_acc


if __name__ == '__main__':
    main()
