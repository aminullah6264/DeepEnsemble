import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class MNISTModel(nn.Module):
    
    def __init__(self) -> None:
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, 5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84, bias=True)
        self.linear2 = nn.Linear(84, 10, bias=True)
        # self.transform = torchvision.transforms.Compose([torchvision.transforms.Normalize([0.1307, ], [0.3081, ])])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.transform(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.linear2(x)
        return x



class MNISTEnsemble(nn.Module):
    def __init__(self, num_ensembles = 10) -> None:
        super(MNISTEnsemble, self).__init__()

        self.models = nn.ModuleList()

        for i in range(num_ensembles):
            self.models.append(MNISTModel())

        self.ensemble_output = None

    def forward(self, x):
        ensemble_output = []
        for model in self.models:
            output = model(x)
            ensemble_output.append(output)

        self.ensemble_output = torch.stack(ensemble_output)
        # import ipdb; ipdb.set_trace()
        # output = torch.mean(self.ensemble_output, 0)
        output = self.ensemble_output.permute(1,0,2)
        
        return output, self.ensemble_output



class FMNISTModel(nn.Module):
    
    def __init__(self) -> None:
        super(FMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, 5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84, bias=True)
        self.linear2 = nn.Linear(84, 10, bias=True)
        # self.transform = torchvision.transforms.Compose([torchvision.transforms.Normalize([0.73, ], [0.90, ])])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.transform(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.linear2(x)
        return x



class FMNISTEnsemble(nn.Module):
    def __init__(self, num_ensembles = 10) -> None:
        super(FMNISTEnsemble, self).__init__()

        self.models = nn.ModuleList()

        for i in range(num_ensembles):
            self.models.append(FMNISTModel())

        self.ensemble_output = None

    def forward(self, x):
        ensemble_output = []
        for model in self.models:
            output = model(x)
            ensemble_output.append(output)

        self.ensemble_output = torch.stack(ensemble_output)
        # import ipdb; ipdb.set_trace()
        # output = torch.mean(self.ensemble_output, 0)
        output = self.ensemble_output.permute(1,0,2)
        
        return output, self.ensemble_output


class CifarModel(nn.Module):
    
    def __init__(self) -> None:
        super(CifarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(256, 128, bias=True)
        self.linear2 = nn.Linear(128, 10, bias=True)
        # self.transform = torchvision.transforms.Compose([torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
        #                                     [0.2023, 0.1994, 0.2010])])



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.transform(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.linear1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.linear2(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1 or self.in_planes != self.planes:
            out_s = self.shortcut_conv(x)
            out += self.shortcut_bn(out_s)
            out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

       
def ResNet18():
    return ResNet(BasicBlock, [3, 3, 3]  , num_classes=10)


class CifarEnsembleRes(nn.Module):
    def __init__(self, num_ensembles = 10) -> None:
        super(CifarEnsembleRes, self).__init__()

        self.models = nn.ModuleList()
        self.ensemble_output = None

        for _ in range(num_ensembles):
            self.models.append(ResNet(BasicBlock, [3, 3, 3], num_classes=10)) 

    def _get_list(self):
        return self.models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensemble_output = []
        for model in self.models:
            output = model(x)
            ensemble_output.append(output)

        self.ensemble_output = torch.stack(ensemble_output)

        output = self.ensemble_output.permute(1,0,2)
        
        return output, self.ensemble_output



class CifarEnsemble(nn.Module):
    def __init__(self, num_ensembles = 10) -> None:
        super(CifarEnsemble, self).__init__()

        self.models = nn.ModuleList()

        for i in range(num_ensembles):
            # self.models.append(CifarModelRes())

            self.models.append(CifarModel())

        self.ensemble_output = None

    def forward(self, x):
        ensemble_output = []
        for model in self.models:
            output = model(x)
            ensemble_output.append(output)

        self.ensemble_output = torch.stack(ensemble_output)

        # import ipdb; ipdb.set_trace()
        # output = torch.mean(self.ensemble_output, 0)
        output = self.ensemble_output.permute(1,0,2)
        
        return output, self.ensemble_output




