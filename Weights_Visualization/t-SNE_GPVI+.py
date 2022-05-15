import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'NF_ResNet')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.manifold import TSNE

import functools

from matplotlib import pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


# set up generator network
def replace_weights_generator(generator, state_dict, keys_w, keys_b):
    generator.linear1.weight.data = state_dict[keys_w[0]]
    generator.linear2.weight.data = state_dict[keys_w[1]]
    generator.linear3.weight.data = state_dict[keys_w[2]]
    generator.linear4.weight.data = state_dict[keys_w[3]]

    # generator.linear1.bias.data = state_dict[keys_b[0]]
    # generator.linear2.bias.data = state_dict[keys_b[1]]
    # generator.linear3.bias.data = state_dict[keys_b[2]]
    # generator.linear4.bias.data = state_dict[keys_b[3]]
    return generator

def extract_parameters(models):
    params = []
    state_dict = {}
    for model in models:
        model_param = torch.tensor([]).to(DEVICE)
        for name, param in model.named_parameters():
            if param.requires_grad: # and 'bn' not in name:
                p = param.view(-1).to(DEVICE)  #.clone().detach()
                start_idx = len(model_param)
                model_param = torch.cat((model_param, p), -1)
                end_idx = len(model_param)
                state_dict[name] = (param.shape, start_idx, end_idx)
        state_dict['param_len'] = len(model_param)
        params.append(model_param)
    params = torch.stack(params)
    return params, state_dict


def insert_parameters(models, item_list, state_dict):
    for i, model in enumerate(models):
        for name, param in model.named_parameters():
            if param.requires_grad:# and 'bn' not in name:                
                shape, start, end = state_dict[name]
                if len(shape) == 0:
                    params_to_model = item_list[i, start:end].view(1)          
                    param.data = params_to_model 
                else:
                    params_to_model = item_list[i, start:end].view(*shape)             
                    param.data = params_to_model
                    


class Generator(nn.Module):
    def __init__(self, h_dim):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(h_dim[0], h_dim[1], bias=True)
        self.linear2 = nn.Linear(h_dim[2], h_dim[3], bias=True)
        self.linear3 = nn.Linear(h_dim[4], h_dim[5], bias=True)
        self.linear4 = nn.Linear(h_dim[6], h_dim[7], bias=False)

    def forward(self, z):
        x = self.linear1(z)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x



# Nonlinearities. Note that we bake the constant into the
# nonlinearites rather than the WS layers.
nonlinearities =    {'silu': lambda x: F.silu(x) / .5595,
                    'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
                    'identity': lambda x: x}



def count_params(module):
    sum([item.numel() for item in module.parameters()])


class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size,
    stride=1, padding=0,
    dilation=1, groups=1, bias=False, gain=False,
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
        kernel_size=1, bias=False)
        self.se_conv1 = nn.Conv2d(width, in_channels,
        kernel_size=1, bias=False)
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
        self.which_conv = functools.partial(ScaledWSConv2d, gain=False, bias=False)

        self.conv1 = self.which_conv(3, 16, kernel_size=3, stride=1, padding=1, gain=False, bias=False)
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


class CifarEnsembleRes(nn.Module):
    def __init__(self, model_weights_path) -> None:
        super(CifarEnsembleRes, self).__init__()

        self.models = nn.ModuleList()
        self.ensemble_output = None


        gpvi_weights_path = model_weights_path
        state_dict = torch.load(gpvi_weights_path, map_location= DEVICE)

          

        weight_keys = list(state_dict.keys())#[::2]       # last two layers are trained model BN mean and var
        bias_keys = list(state_dict.keys())#[1::2]

        weight_size = []

        for k in weight_keys:
            weight_size.extend(state_dict[k].shape[::-1])
       

        assert len(weight_size) == 8, "should be 8 sizes of weights for 4 layers"
        generator = Generator(weight_size).to(DEVICE)

        generator = replace_weights_generator(generator, state_dict, weight_keys, bias_keys)
        input_size = weight_size[0]
        output_size = weight_size[-1]
        for _ in range(10):
            self.models.append(NF_ResNet(NF_BasicBlock, [3, 3, 3], num_classes=10)) 
            

        param_list, state_dict = extract_parameters(self.models)
        input_noise = torch.randn(len(self.models), input_size).to(DEVICE)
        sampled_weights = generator(input_noise)

        # import ipdb; ipdb.set_trace()
        print( param_list.shape, sampled_weights.shape)
        param_list = sampled_weights

        insert_parameters(self.models, param_list, state_dict)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensemble_output = []
        for model in self.models:
            output = model(x)
            ensemble_output.append(output)

        self.ensemble_output = torch.stack(ensemble_output)

        # import ipdb; ipdb.set_trace()
        # output = torch.mean(self.ensemble_output, 0)
        output = self.ensemble_output.permute(1,0,2)
        
        return output, self.ensemble_output



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

NUM_TRAJECTORIES = 10


prob_ = []
for i in range(40):
  model_weights_path = '/nfs/stak/users/ullaham/hpc-share/Adv/GPVIPlus_Updated/stochastic_parvi_GPVI+NF_ResNet/Normalize0_1%255/Standard_NF_ResNet_OpenSet_Stochastic_PaVI_1e-5_Entropy_1e-3_Cifar6_Test_Checkpoints_' +str(i) + 'GPVI+/generator.pt'

  EnsembleNet = CifarEnsembleRes(model_weights_path).to(DEVICE)

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
print('[INFO] shape of predictions tensor: ', predictions_for_tsne.shape)
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

plt.savefig('GPVI+_loss_landscape_2d.png')


