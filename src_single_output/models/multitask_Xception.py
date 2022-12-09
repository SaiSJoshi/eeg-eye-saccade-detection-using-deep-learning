import torch
import torch.nn as nn
import numpy as np
import math


class ResBlock_Xception(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shortcut = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0),
                             max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0),
                             max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, groups=self.in_channels,
                      kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0),
                             max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, groups=self.out_channels,
                      kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.out_channels,
                      out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=self.out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0),
                             max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, groups=self.out_channels,
                      kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=self.out_channels,
                      out_channels=self.out_channels, kernel_size=1),
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
    def __init__(self, input_shape, output_LR, output_Angle, output_Amp, output_Pos, kernel_size=40, nb_filters=64, depth=6, batch_size=64):
        super().__init__()
        self.timesamples = input_shape[1]
        self.in_channels = input_shape[0]
        
        self.output_LR = output_LR
        self.output_Angle = output_Angle
        self.output_Amp = output_Amp
        self.output_Pos = output_Pos
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.depth = depth
        self.batch_size = batch_size
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = nn.ConstantPad1d(padding=(0, 1), value=0)
        self.output_layer_LR = nn.Sequential(
            nn.Linear(in_features=self.nb_filters *
                      self.timesamples, out_features=output_LR)
        )
        self.output_layer_Angle = nn.Sequential(
            nn.Linear(in_features=self.nb_filters *
                      self.timesamples, out_features=output_Angle)
        )
        self.output_layer_Amp = nn.Sequential(
            nn.Linear(in_features=self.nb_filters *
                      self.timesamples, out_features=output_Amp)
        )
        self.output_layer_Pos = nn.Sequential(
            nn.Linear(in_features=self.nb_filters *
                      self.timesamples, out_features=output_Pos)
        )

        
        modules = []
        for d in range(self.depth):
            if d == 0:
                modules.append(ResBlock_Xception(in_channels=self.in_channels, out_channels=self.nb_filters, kernel_size=self.kernel_size))
            else:
                modules.append(ResBlock_Xception(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size))
        
        self.ResBlock = nn.Sequential(*modules)


    def forward(self, x, return_feats=False):
        tmp = self.ResBlock(x)
        tmp = self.gap_layer_pad(tmp)
        tmp = self.gap_layer(tmp)
        tmp = tmp.view(tmp.size(0), -1)  # flatten

        out_LR = torch.nan_to_num(self.output_layer_LR(tmp)) # This is the output for L/R task
        out_Angle = torch.nan_to_num(self.output_layer_Angle(tmp)) # Output for Angle task
        out_Amp = torch.nan_to_num(self.output_layer_Amp(tmp)) # Output for Ampitude task
        out_Pos = torch.nan_to_num(self.output_layer_Pos(tmp))
        
        if return_feats:
            return tmp
        else:
            return out_LR, out_Angle, out_Amp, out_Pos
