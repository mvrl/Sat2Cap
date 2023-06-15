import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GeoNet(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64,fc_dim)
       

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output

if __name__ == '__main__':
    net = GeoNet(fc_dim=512)
    x = torch.randn(2,12)
    print(net(x).shape) #torch.Size([2, fc_dim])
    