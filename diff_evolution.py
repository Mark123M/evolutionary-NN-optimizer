import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import math
import random
import copy
import sys
import time
from IPython.display import clear_output

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.manual_seed(1122)

theta = [0.1, 1, 1.8, 2]

def gaussian(x, mu):
    return (1 / (0.3 * math.sqrt(2 * math.pi))) * (math.e ** ((-1/2) * (((x - mu) / 0.3)) ** 2))

def gaussian_mixture(x):
    return gaussian(x, theta[0]) + gaussian(x, theta[1]) + gaussian(x, theta[2]) + gaussian(x, theta[3])

class DE_NN(nn.Module):
    def __init__(self, NP, CR, F):
        super(DE_NN, self).__init__()
        lin1s = nn.ModuleList([nn.Linear(1, 4) for i in range(NP)])
        lin2s = nn.ModuleList([nn.Linear(4, 8) for i in range(NP)])
        lin3s = nn.ModuleList([nn.Linear(8, 4) for i in range(NP)])
        lin4s = nn.ModuleList([nn.Linear(4, 1) for i in range(NP)])
        self.layers = nn.ModuleList([lin1s, lin2s, lin3s, lin4s])
        self.layers_len = len(self.layers)
        self.NP = NP
        self.CR = CR
        self.F = F
        self.min_l = float('inf')
        self.best_model = 0
    def forward_i(self, X, layers): # a single pass
        for k in range(self.layers_len - 1):
            X = torch.relu(layers[k](X))
        return layers[self.layers_len - 1](X)
    def forward(self, X):
        for k in range(self.layers_len - 1):
            X = torch.relu(self.layers[k][self.best_model](X))
        return self.layers[self.layers_len - 1][self.best_model](X)  
    def step(self, id, X, Y, L, type='param'): # forward pass with candidate i
        fx = L(self.forward_i(X, [l[id] for l in self.layers]), Y)
        agent_ids = random.sample(range(0, self.NP), 3) # how to efficiently reject self? rej sampling?
        y = [copy.deepcopy(l[id]).to(device) for l in self.layers]

        if type == 'param':
            for k in range(self.layers_len):
                R = tuple(random.randint(0, dim) for dim in self.layers[k][id].weight.shape)
                for i in self.idxs[k]:
                    ri = random.random()
                    if ri < self.CR or i == R:
                        y[k].weight[i] = self.layers[k][agent_ids[0]].weight[i] + self.F * (self.layers[k][agent_ids[1]].weight[i] - self.layers[k][agent_ids[2]].weight[i])
                    else:
                        y[k].weight[i] = self.layers[k][id].weight[i]
        elif type == 'block':
            R = random.randint(0, self.layers_len)
            for i in range(self.layers_len):
                ri = random.random()
                if ri < self.CR or i == R:
                    y[i].weight = torch.nn.Parameter(self.layers[i][id].weight + self.F * (self.layers[i][self.best_model].weight - self.layers[i][id].weight)
                                                     + self.F * (self.layers[i][agent_ids[0]].weight - self.layers[i][agent_ids[1]].weight))
                    y[i].bias = torch.nn.Parameter(self.layers[i][id].bias + self.F * (self.layers[i][self.best_model].bias - self.layers[i][id].bias)
                                    + self.F * (self.layers[i][agent_ids[0]].bias - self.layers[i][agent_ids[1]].bias))
                else:
                    y[i].weight = torch.nn.Parameter(self.layers[i][id].weight)
                    y[i].bias = torch.nn.Parameter(self.layers[i][id].bias)
        
        y[i].weight *= 0.99
        fy = L(self.forward_i(X, y), Y)

        if fy <= fx: 
            for k in range(self.layers_len):
                self.layers[k][id] = y[k]
            fx = fy
        if fx < self.min_l:
            self.best_model = id
            self.min_l = fx

epochs = 2000

X = torch.rand(40000, 1).to(device) * 5 - 1
Y = gaussian_mixture(X).to(device)

num_params = int(89 / 2)
NP = num_params
CR = 0.9
F = 0.8


model = DE_NN(NP, CR, F).to(device) 
model = torch.compile(model, mode='default')
L = nn.MSELoss().to(device)

print(sys.argv[1])
#with torch.autograd.profiler.profile(use_device='cuda') as prof:
if sys.argv[1] == 'nsight':
    with torch.no_grad():
        for i in range(model.NP):
            model.step(i, X, Y, L, 'block')
        print(model.min_l)
elif sys.argv[1] == 'chrome':
    with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as p:
        with torch.no_grad():
            for i in range(model.NP):
                model.step(i, X, Y, L, 'block')
            print(model.min_l)
    prefix = f"{int(time.time())}"
    p.export_chrome_trace(f"chrome_trace_{prefix}.json.gz")
    p.export_memory_timeline(f"chrome_memory_")
