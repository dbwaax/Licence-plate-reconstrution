import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),        
        )
        self.shortcut = nn.Sequential()
        if(stride!=1 or out_channels!=in_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out 