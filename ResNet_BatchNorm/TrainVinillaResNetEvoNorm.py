from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import  ResNet18
from utils import classification_loss, _classification_vote
# import torchsummary

dataset = 'cifar10'  # can be 'mnist', 'f_mnist', 'cifar10'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 200


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

Net = ResNet18()

Net.to(DEVICE)

lr_ = 0.1
momentum_ = 0.9
wd_ = 1e-4
optimizer = torch.optim.SGD(Net.parameters(), lr_,
                            momentum= momentum_,
                            weight_decay= wd_)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[82, 123], last_epoch= -1 )
lossFun = nn.CrossEntropyLoss()

for epoch in range(1, epochs):
    train_loss = 0
    train_correct = 0

    Net.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()

        output = Net(data)
        loss = lossFun(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item()
        

    lr_scheduler.step()

    train_loss /= len(train_loader)
    print('Epoch: {} ResNet-18 Vinilla BatchNorm--> {} Dataset Training Loss = {:.4f}, Train Accuracy =  {:.2f}%\n'.format(
        epoch, dataset, train_loss, 100. * train_correct / (len(train_loader.dataset))))

    Net.eval()

    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = Net(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        print('Epoch: {} , Test Accuracy =  {:.2f}%\n'.format(
            epoch, 100. * correct / len(test_loader.dataset)))

    
    

   