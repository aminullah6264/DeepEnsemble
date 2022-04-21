import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

import numpy as np



# Nonlinearities. Note that we bake the constant into the
# nonlinearites rather than the WS layers.
nonlinearities =    {'silu': lambda x: F.silu(x) / .5595,
                    'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
                    'identity': lambda x: x}

# Block base widths and depths for each variant
params = {'NF-ResNet-18': {'width': [16, 32, 64],  'depth': [3, 3, 3],
            'train_imsize': 32, 'test_imsize': 224,
            'weight_decay': 2e-5, 'drop_rate': 0.2},
        
            }

def count_params(module):
    sum([item.numel() for item in module.parameters()])


class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size,
    stride=1, padding=0,
    dilation=1, groups=1, bias=True, gain=True,
    eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels,
        kernel_size, stride,
        padding, dilation,
        groups, bias)
        if gain:
            self.gain = nn.Parameter(
            torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        # Epsilon, a small constant to avoid dividing by zero.
        self.eps = eps
    def get_weight(self):
        # Get Scaled WS weight OIHW;
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3],
        keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3],
        keepdims=True)
        weight = (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        if self.gain is not None:
            weight = weight * self.gain
        return weight
    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias,
        self.stride, self.padding,
        self.dilation, self.groups)


class SqueezeExcite(nn.Module):
    """Simple Squeeze+Excite layers."""
    def __init__(self, in_channels, width, activation):
        super().__init__()
        self.se_conv0 = nn.Conv2d(in_channels, width,
        kernel_size=1, bias=True)
        self.se_conv1 = nn.Conv2d(width, in_channels,
        kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x):
        # Mean pool for NCHW tensors
        h = torch.mean(x, axis=[2, 3], keepdims=True)
        # Apply two linear layers with activation in between
        h = self.se_conv1(self.activation(self.se_conv0(h)))
        # Rescale the sigmoid output and return
        return (torch.sigmoid(h) * 2) * x




class NF_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=F.relu, which_conv=ScaledWSConv2d,
    beta=1.0, alpha=1.0, se_ratio=0.5):
        super(NF_BasicBlock, self).__init__()

        self.activation = activation
        self.beta, self.alpha = beta, alpha

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = which_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = which_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        

        if stride != 1 or in_planes != planes:
            self.shortcut_conv = which_conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride)
        
        self.se = SqueezeExcite(self.expansion * planes, self.expansion * planes, self.activation)
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        out = self.activation(x) / self.beta
        if self.stride != 1 or self.in_planes != self.planes:
            shortcut = self.shortcut_conv(out)
        else:
            shortcut = x
        out = self.conv1(out) # Initial bottleneck conv
        out = self.conv2(self.activation(out)) # Spatial conv
        out = self.se(out) # Apply squeeze + excite to middle block.

        return out * self.skipinit_gain * self.alpha + shortcut
        
      


class NF_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, se_ratio=0.5, alpha=0.2, 
    activation='silu', drop_rate=None, stochdepth_rate=0.0):
        super(NF_ResNet, self).__init__()
        self.in_planes = 16
        self.se_ratio = se_ratio
        self.alpha = alpha
        self.activation = nonlinearities.get(activation)
        self.stochdepth_rate = stochdepth_rate
        self.which_conv = functools.partial(ScaledWSConv2d, gain=True, bias=True)

        self.conv1 = self.which_conv(3, 16, kernel_size=3, stride=1, padding=1)
        expected_var = 1.0
        beta = expected_var ** 0.5
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio)
        expected_var += self.alpha ** 2
        beta = expected_var ** 0.5
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio)

        expected_var += self.alpha ** 2
        beta = expected_var ** 0.5
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio)
        
        self.linear = nn.Linear(64, num_classes, bias=True)
        torch.nn.init.zeros_(self.linear.weight)

    def _make_layer(self, block, planes, num_blocks, stride, activation=F.relu, which_conv=ScaledWSConv2d,
    beta=1.0, alpha=1.0, se_ratio=0.5):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation, which_conv=which_conv,
    beta=beta, alpha=alpha, se_ratio=se_ratio))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

       
def NF_ResNet18():
    return NF_ResNet(NF_BasicBlock, [3, 3, 3]  , num_classes=10)




class CifarEnsembleRes(nn.Module):
    def __init__(self, num_ensembles = 10) -> None:
        super(CifarEnsembleRes, self).__init__()

        self.models = nn.ModuleList()
        self.ensemble_output = None

        for _ in range(num_ensembles):
            self.models.append(NF_ResNet(NF_BasicBlock, [3, 3, 3], num_classes=10)) 

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


model = NF_ResNet18().to('cuda')
from torchsummary import summary
summary(model, (3, 32, 32))
