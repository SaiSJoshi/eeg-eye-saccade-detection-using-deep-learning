import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import math

class ResBlock(nn.Module):
    def  __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shortcut = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0), max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, groups = self.in_channels,
            kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0), max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, groups = self.out_channels,
            kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0), max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, groups = self.out_channels,
            kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0), max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, groups = self.out_channels,
            kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.activation = nn.ReLU()

    def forward(self, input):
        shortcut = self.shortcut(input)
        tmp = self.activation(self.conv1(input))
        tmp = self.activation(self.conv2(tmp))
        tmp = self.activation(self.conv3(tmp))
        output = self.activation(tmp+shortcut)
        
        return output

class Xception(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=40, nb_filters=64, depth = 6, batch_size = 64):
        super().__init__()
        self.timesamples = input_shape[0]
        self.in_channels = input_shape[1]
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.depth = depth
        self.batch_size = batch_size
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = nn.ConstantPad1d(padding=(0, 1), value=0)
        self.output_layer = nn.Sequential(
                nn.Linear(in_features=self.nb_filters * self.timesamples, out_features=output_shape) 
            )
    
    def forward(self, x, return_feats=False):
        
        for d in range(self.depth):
            if d == 0:
                tmp = ResBlock(in_channels = self.in_channels, out_channels = self.nb_filters, kernel_size = self.kernel_size)(x)
            else:
                tmp = ResBlock(in_channels = self.nb_filters, out_channels = self.nb_filters, kernel_size = self.kernel_size)(tmp)
        
        tmp = self.gap_layer_pad(tmp) 
        tmp = self.gap_layer(tmp)
        tmp = tmp.view(self.batch_size, -1) # flatten
        output = self.output_layer(tmp)

        if return_feats:
            return tmp
        else:
            return output