import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import math
import torch.profiler
import sys
import time
import torch.cuda.nvtx as nvtx
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.manual_seed(1122)

theta = [0.1, 1, 1.8, 2]

def gaussian(x, mu):
    return (1 / (0.3 * math.sqrt(2 * math.pi))) * (math.e ** ((-1/2) * (((x - mu) / 0.3)) ** 2))

def gaussian_mixture(x):
    return gaussian(x, theta[0]) + gaussian(x, theta[1]) + gaussian(x, theta[2]) + gaussian(x, theta[3])

class SmallNN(nn.Module):
    def __init__(self):
        super(SmallNN, self).__init__()
        self.lin1 = nn.Linear(1, 4)
        self.lin2 = nn.Linear(4, 8)
        self.lin3 = nn.Linear(8, 4)
        self.lin4 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        return self.lin4(x)

model = SmallNN().to(device)
model = torch.compile(model, mode='default')
L = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10

X = torch.rand(400000, 1).to(device) * 5 - 1
Y = gaussian_mixture(X).to(device)

if sys.argv[1] == 'nsight':
    for e in range(epochs):
        nvtx.range_push(f"Epoch {e}")
        nvtx.range_push("forward_pass")
        y_pred = model(X)
        loss = L(y_pred, Y)
        nvtx.range_pop()
        nvtx.range_push("backward_prop")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()
        print(loss.item())
        nvtx.range_pop()
elif sys.argv[1] == 'chrome':
    with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=False) as p:
        for e in range(epochs):
            y_pred = model(X)
            loss = L(y_pred, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    prefix = f"Adam_{int(time.time())}"
    p.export_chrome_trace(f"chrome_trace_{prefix}.json.gz")

