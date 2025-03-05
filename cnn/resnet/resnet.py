import torch
import torch.nn as nn   
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.exanpsion = 4
        self.conv1 = nn. Conv2d(in_channels,out_channels,kernal_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn. Conv2d(in_channels,out_channels,kernal_size=3,stride=stride,padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn. Conv2d(in_channels,out_channels*self.exanpsion,kernal_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.exanpsion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        # this would be a conv layer that we are going to do to the identity mapping
        # so that the dimensions match up

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


    