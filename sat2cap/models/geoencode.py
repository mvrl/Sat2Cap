import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import code
#from ..utils.locationencoder.locationencoder.locationencoder import LocationEncoder
from .siren import LocationEncoder

#encoder for data and time. We use linear layers for this
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

#this is the spherical harmonics encoder
class SphericalNet(nn.Module):
    def __init__(self, fc_dim = 512, use_time=True):
        super().__init__()
        hparams = dict(
        legendre_polys=30,
        dim_hidden=512,
        num_layers=4,
        num_classes=fc_dim
    )
        self.location_encoder = LocationEncoder('sphericalharmonics', 'siren', hparams)
        self.time_encoder = TimeNet()

    def forward(self, x):
       # 
        lonlat = x[:,:2]
        date_time = x[:,2:]
        lonlat_encoding = self.location_encoder(lonlat)
        date_time_encoding = self.time_encoder(date_time)

        meta_encoding = lonlat_encoding + date_time_encoding

        return meta_encoding

class TimeNet(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64,fc_dim)
       

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output



class DateNet(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)

        self.fc2 = nn.Linear(256,512)

        self.fc3 = nn.Linear(512,128)
       

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output

class ShallowNet(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256,128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = F.relu(self.fc2(x))
        return output

class PlacesHead(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,365)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class TransientHead(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,40)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        return output

class ResNetHead(nn.Module):
    def __init__(self,fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(2048, 1000)

        self.fc2 = nn.Linear(1000, fc_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output




if __name__ == '__main__':
    # net = GeoNet(fc_dim=512)
    # x = torch.randn(2,12)
    # print(net(x).shape) #torch.Size([2, fc_dim])
    net = SphericalNet()
    x = torch.randn(2,10)
    out = net(x)
    print(out.shape)