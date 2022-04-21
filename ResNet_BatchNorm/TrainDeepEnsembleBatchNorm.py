import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import MNISTEnsemble, CifarEnsemble, FMNISTEnsemble, CifarEnsembleRes
from utils import classification_loss, _classification_vote
# import torchsummary

dataset = 'cifar10'  # can be 'mnist', 'f_mnist', 'cifar10'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 200
# How many Ensemble model you want to create num_ensembles = X
num_ensembles = 10


if dataset =='mnist':
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.1307, ], [0.3081, ])])

    dataset1 = torchvision.datasets.MNIST('MnistData', train=True, download=True,
                        transform=transform)
    dataset2 = torchvision.datasets.MNIST('MnistData', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=5000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=5000)

    EnsembleNet = MNISTEnsemble(num_ensembles = num_ensembles).to(DEVICE)


if dataset =='f_mnist':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize([0.73, ], [0.90, ])
        ])

    dataset1 = torchvision.datasets.FashionMNIST('F_MnistData', train=True, download=True,
                        transform=transform)
    dataset2 = torchvision.datasets.FashionMNIST('F_MnistData', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=2000,  shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=2000)

    EnsembleNet = FMNISTEnsemble(num_ensembles = num_ensembles).to(DEVICE)



if dataset =='cifar10':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])
                                            ])

    dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True, 
                                                    transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                transform ])   )
                
    dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=500)

    EnsembleNet = CifarEnsembleRes(num_ensembles = num_ensembles).to(DEVICE)



lr_ = 0.1
momentum_ = 0.9
wd_ = 1e-4
optimizer = torch.optim.SGD(EnsembleNet.parameters(), lr_,
                            momentum= momentum_,
                            weight_decay= wd_)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[82, 123], last_epoch= -1 )


# optimizer_list = [torch.optim.SGD(m.parameters(), lr=lr_, momentum= momentum_, weight_decay= wd_) for m in
#                       EnsembleNet._get_list()]  

# lr_scheduler_list = [torch.optim.lr_scheduler.MultiStepLR(opt,
#                                                 milestones=[82, 123], last_epoch= -1 ) for opt in
#                     optimizer_list]  


Best_Acc = 0
for epoch in range(1, epochs):
    train_loss = 0
    train_correct = []

    EnsembleNet.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        targets = target.unsqueeze(1).expand(*target.shape[:1], num_ensembles,
                                            *target.shape[1:])
        
        optimizer.zero_grad()
        # for opt in optimizer_list:
        #     opt.zero_grad()
        output, _ = EnsembleNet(data)
          
        loss, avg_acc = classification_loss(output, targets)

        loss.backward()
        optimizer.step()
        
        # for opt in optimizer_list:
        #     opt.step()

        train_correct.append(avg_acc.item())
        train_loss += loss.item()
    
    # for lr_s in lr_scheduler_list:
    #     lr_s.step()
    lr_scheduler.step()
    train_loss /= len(train_loader)
    train_acc = torch.as_tensor(train_correct).mean() * 100
    print('Epoch: {} ResNet-18 DeepEnsemble_BatchNorm--> {} Dataset Training Loss = {:.4f}, Train Accuracy =  {:.2f}%\n'.format(
        epoch, dataset, train_loss, train_acc))

    EnsembleNet.eval()

    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output, _ = EnsembleNet(data)
            _, acc = _classification_vote(output, target)
            correct += acc.item()
        
        print('Epoch: {} , DeepEnsemble_BatchNorm Test Accuracy =  {:.2f}%\n'.format(
            epoch, 100. * correct / len(test_loader.dataset)))


    if train_acc > Best_Acc:
        modelName =  dataset + 'DeepEnsemble_BatchNorm.pt'
        torch.save(EnsembleNet.state_dict(), modelName)
        Best_Acc = train_acc