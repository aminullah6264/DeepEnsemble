

import torch
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import sys, os
import functools
sys.path.append(os.path.abspath(os.path.join('..', 'NF_ResNet')))
import seaborn as sns
from matplotlib import pyplot as plt

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
            # confidence = pred.reshape(-1)


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


# set up generator network
def replace_weights_generator(generator, state_dict, keys_w, keys_b):
    generator.linear1.weight.data = state_dict[keys_w[0]]
    generator.linear2.weight.data = state_dict[keys_w[1]]
    generator.linear3.weight.data = state_dict[keys_w[2]]
    generator.linear4.weight.data = state_dict[keys_w[3]]

    generator.linear1.bias.data = state_dict[keys_b[0]]
    generator.linear2.bias.data = state_dict[keys_b[1]]
    generator.linear3.bias.data = state_dict[keys_b[2]]
    generator.linear4.bias.data = state_dict[keys_b[3]]
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
        self.linear4 = nn.Linear(h_dim[6], h_dim[7], bias=True)

    def forward(self, z):
        x = self.linear1(z)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x



model_weights_path = '/nfs/stak/users/ullaham/hpc-share/Adv/GPVIPlus_Updated/stochastic_parvi_GPVI+NF_ResNet/Normalize0_1%255/Standard_NF_ResNet_OpenSet_Stochastic_PaVI_1e-5_Entropy_1e-3_Cifar6_V2_GPVI+/generator.pt'

gpvi_weights_path = model_weights_path
state_dict = torch.load(gpvi_weights_path, map_location= DEVICE)

  

weight_keys = list(state_dict.keys())[::2][:-1]         # last two layers are trained model BN mean and var
bias_keys = list(state_dict.keys())[1::2][:-1]

weight_size = []

for k in weight_keys:
    weight_size.extend(state_dict[k].shape[::-1])


assert len(weight_size) == 8, "should be 8 sizes of weights for 4 layers"
generator = Generator(weight_size).to(DEVICE)

generator = replace_weights_generator(generator, state_dict, weight_keys, bias_keys)
input_size = weight_size[0]
output_size = weight_size[-1]

num_particles = 50
input_noise = torch.randn(num_particles, input_size).to(DEVICE)
params_ = generator(input_noise).detach().cpu().numpy()

print('Params shape', params_.shape)

empty_arr = np.zeros(shape=(num_particles,num_particles))

for i in range(num_particles):
  weights1 = params_[i]
  for j in range(i, num_particles):
    weights2 = params_[j]
    
    # compute cosine similarity of weights
    cos_sim = np.dot(weights1, weights2)/(norm(weights1)*norm(weights2))
    
    empty_arr[i][j] = cos_sim
    if i is not j:
      empty_arr[j][i] = cos_sim

cos_sim_coeff = empty_arr[::-1]


plt.figure(figsize=(10,8))
sns.heatmap(cos_sim_coeff, cmap='RdBu_r');
plt.xticks(np.arange(0, num_particles, step=1), np.arange(0, num_particles, 1));
plt.yticks(np.arange(0, num_particles, step=1), np.arange(num_particles, 0, -1));

plt.savefig('GPVI+independent_functional_similarity.png')