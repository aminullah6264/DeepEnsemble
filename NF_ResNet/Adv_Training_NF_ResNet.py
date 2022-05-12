import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'torchattacks')))
from torchattacks import PGD, PGDL2, FGSM, DeepFool, MultiAttack, MIFGSM, RFGSM
from tqdm import tqdm
from nf_resnet import CifarEnsembleRes
from utils import classification_loss, _classification_vote
# import torchsummary

dataset = 'cifar10'  # can be 'mnist', 'f_mnist', 'cifar10'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attack = 'PGD'                 # PGD, FGSM, IFGSM
eps_ = 0.031
eps_alpha_ = 0.007
steps_ = 7

attackNorm = 'Linf'           # L2, Linf

# attack = 'FGSM'                 # PGD, FGSM, IFGSM
# eps_ = 128/255
# eps_alpha_ = 16/255
# steps_ = 7

epochs = 100
# How many Ensemble model you want to create num_ensembles = X
num_ensembles = 10


if dataset =='cifar10':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
                                            ])

    dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True,
                                            transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                transform ]))    
    dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=512)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=500)

    # EnsembleNet = CifarEnsemble(num_ensembles = num_ensembles).to(DEVICE)
    EnsembleNet = CifarEnsembleRes(num_ensembles = num_ensembles).to(DEVICE)


if attackNorm == 'Linf':
    if attack == 'PGD':
        attackk = PGD(EnsembleNet, eps= eps_, alpha=  eps_alpha_, steps =  steps_)
    if attack == 'FGSM':
        attackk = FGSM(EnsembleNet, eps=  eps_)
    if attack == 'IFGSM':
        attackk = PGD(EnsembleNet, eps= eps_, alpha=  eps_alpha_, steps =  steps_, ensemble=True, num_ensembles= num_ensembles, random_start=False)

if attackNorm == 'L2':
    if attack == 'PGD':
        attackk = PGDL2(EnsembleNet, eps= eps_, alpha= eps_alpha_, steps = steps_, ensemble=True, num_ensembles = num_ensembles)

    if attack == 'FGSM': # when eps and alpha are equal, steps = 1, and eps_for_division=1e-10 PGDL2 works as FGSML2
        attackk = PGDL2(EnsembleNet, eps= eps_, alpha= eps_, steps = 1, eps_for_division=1e-10, ensemble=True, num_ensembles = num_ensembles)
    
    if attack == 'IFGSM':
        attackk = PGDL2(EnsembleNet, eps= eps_, alpha= eps_alpha_, steps = steps_, random_start= False, ensemble=True, num_ensembles = num_ensembles)


lr_ = 0.1
momentum_ = 0.9
wd_ = 1e-4
optimizer = torch.optim.SGD(EnsembleNet.parameters(), lr_,
                            momentum= momentum_,
                            weight_decay= wd_)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[82, 123], last_epoch= -1 )

Best_Acc = 0
for epoch in range(1, epochs):
    train_loss = 0
    train_correct = []

    EnsembleNet.train()
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        Adv_data = attackk(data, target).to(DEVICE)
        
        data = torch.cat((data, Adv_data),0).to(DEVICE)
        target = torch.cat((target, target),0).to(DEVICE)

        shuffle_idx = np.random.choice(data.shape[0], data.shape[0], replace = False)
        shuffle_idx = torch.from_numpy(shuffle_idx).type(torch.LongTensor)
        data = data[shuffle_idx]
        target = target[shuffle_idx]
        


        targets = target.unsqueeze(1).expand(*target.shape[:1], num_ensembles,
                                            *target.shape[1:])
          

        optimizer.zero_grad()
        output, _ = EnsembleNet(data)
                
        loss, avg_acc = classification_loss(output, targets)

        loss.backward()
        optimizer.step()
        train_correct.append(avg_acc.item())
        train_loss += loss.item()


        
    lr_scheduler.step()
    train_loss /= len(train_loader)
    train_acc = torch.as_tensor(train_correct).mean() * 100

    print('Epoch: {} --> {} dataset, {}  ADV Training Loss = {:.4f}, ADV Train Accuracy =  {:.2f}%\n'.format(
        epoch, dataset, attack, train_loss, train_acc))

    EnsembleNet.eval()
    
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        Adv_data = attackk(data, target).to(DEVICE)
        with torch.no_grad():
            output, _ = EnsembleNet(Adv_data)
            _, acc = _classification_vote(output, target)
            correct += acc.item()
        
    print('Epoch: {}  {} ADV Test Accuracy =  {:.2f}%\n'.format(
        epoch, attackNorm, 100. * correct / len(test_loader.dataset)))

    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output, _ = EnsembleNet(data)
            _, acc = _classification_vote(output, target)
            correct += acc.item()
        
        print('Epoch: {} , Clean Test Accuracy =  {:.2f}%\n'.format(
            epoch, 100. * correct / len(test_loader.dataset)))


    if train_acc > Best_Acc:
        modelName =  dataset + 'BatchNorm' + attack + '_' + attackNorm + '_' + str(eps_) + '_' + str(eps_alpha_) + '_steps_' + str(steps_) + '_.pt'
        torch.save(EnsembleNet.state_dict(), modelName)
        Best_Acc = train_acc
