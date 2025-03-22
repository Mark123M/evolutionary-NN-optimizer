import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.cuda.nvtx as nvtx
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

batch_size = 400000

class DE_NN(nn.Module):
    def __init__(self, NP, CR, F):
        super(DE_NN, self).__init__()
        self.lin1s = torch.randn((NP, 4, 1), requires_grad=False).to(device)
        self.lin2s = torch.randn((NP, 8, 4), requires_grad=False).to(device)
        self.lin3s = torch.randn((NP, 4, 8), requires_grad=False).to(device)
        self.lin4s = torch.randn((NP, 1, 4), requires_grad=False).to(device)
        self.layers_len = 4
        self.NP = NP
        self.CR = CR
        self.F = F
        self.min_l = float('inf')
        self.best_model = 0
    def forward_all(self, X):
        # This is just bmm???
        X = torch.relu(torch.einsum('lik,lkj->lij', self.lin1s, X))
        X = torch.relu(torch.einsum('lik,lkj->lij', self.lin2s, X))

        #M = torch.empty((NP, 8, batch_size)).to(device) # l, i, j
        #for l in range(NP):
        #    for i in range(8):
        #        for j in range(batch_size):
        #            total = 0
        #            for k in range(4):
        #                total += self.lin2s[l,i,k] * X[l,k,j]
        #            M[l,i,j] = total
        #print(torch.sum(torch.relu(M)))

        X = torch.relu(torch.einsum('lik,lkj->lij', self.lin3s, X))
        return torch.einsum('lik,lkj->lij', self.lin4s, X)
    def forward_i(self, X, layers): # a single pass
        for k in range(self.layers_len - 1):
            X = torch.relu(layers[k](X))
        return layers[self.layers_len - 1](X)
    def forward(self, X):
        for k in range(self.layers_len - 1):
            X = torch.relu(self.layers[k][self.best_model](X))
        return self.layers[self.layers_len - 1][self.best_model](X)
    def step(self, X, Y, L, type='param'): # forward pass with candidate i
        nvtx.range_push("forward_1")
        fx = L(self.forward_all(X), Y).mean(dim = 2)
        nvtx.range_pop()
        agent_ids = random.sample(range(0, self.NP), 3) # how to efficiently reject self? rej sampling?
        nvtx.range_push(f"copy layers")
        y1s = torch.randn_like(self.lin1s).to(device, non_blocking=True)
        y2s = torch.randn_like(self.lin2s).to(device, non_blocking=True)
        y3s = torch.randn_like(self.lin3s).to(device, non_blocking=True)
        y4s = torch.randn_like(self.lin4s).to(device, non_blocking=True)
        print(agent_ids, fx.shape, y1s.shape, y2s.shape, y3s.shape, y4s.shape)
        nvtx.range_pop()

epochs = 2000

NP = 44
CR = 0.9
F = 0.8
X = torch.rand(1, batch_size).to(device) * 5 - 1
Y = gaussian_mixture(X).to(device)

print(X.shape, Y.shape)
X = X.unsqueeze(0).expand(NP, 1, batch_size)
Y = Y.unsqueeze(0).expand(NP, 1, batch_size)
print(X.shape, Y.shape)

model = DE_NN(NP, CR, F).to(device) 
model = torch.compile(model, mode='default')
L = nn.MSELoss(reduction='none').to(device)

Y_pred = model.forward_all(X)
print(Y_pred.shape)

print(sys.argv[1])
if sys.argv[1] == 'nsight':
    with torch.no_grad():
        model.step(X, Y, L, 'block')
        print(model.min_l)
elif sys.argv[1] == 'chrome':
    with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as p:
        with torch.no_grad():
            for i in range(model.NP):
                model.step(i, X, Y, L, 'block')
            print(model.min_l)
    prefix = f"{int(time.time())}"
    p.export_chrome_trace(f"chrome_trace_{prefix}.json.gz")
    p.export_memory_timeline(f"chrome_memory_{prefix}.html")
else:
    pass
