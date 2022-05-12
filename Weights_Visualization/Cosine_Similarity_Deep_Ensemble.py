

import torch
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



pretrainedParams = torch.load('../NF_ResNet/cifar10DeepEnsemble_OpenSet_NF_ResNet_auc_entropyCifar6.pt')

pretrainedweights = []
for key in pretrainedParams:
    pretrainedParams[key]
    # print(key, pretrainedParams[key].shape)
    pretrainedweights.append(pretrainedParams[key].reshape(-1))  

params_ = torch.cat(pretrainedweights).reshape(10,-1).cpu()
print(params_.shape)

num_particles = 10

empty_arr = np.zeros(shape=(10,10))

for i in range(10):
  weights1 = params_[i]
  for j in range(i, 10):
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

plt.savefig('DeepEnsembleindependent_functional_similarity.png')