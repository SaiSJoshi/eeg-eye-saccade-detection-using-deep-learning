import torch
import torch.nn as nn
import numpy as np
import math

class ResBlock_PyramidalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,curr_depth,nb_filters=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.curr_depth=curr_depth
        self.nb_filters = nb_filters

               
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((max(math.floor(kernel_size/2)-1, 0),
                             max(math.floor(kernel_size/2), 0)), value=0),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=self.kernel_size,bias=False),
            nn.BatchNorm1d(num_features=(self.curr_depth+1)*self.nb_filters),
            
        )
        self.activation = nn.ReLU()
       

    def forward(self, input):
        output = self.conv1(input)
        output=self.activation(output)
        return output


class PyramidalCNN(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=16, nb_filters=16, depth=6, batch_size=64):
        super().__init__()

        self.timesamples = input_shape[1]
        self.in_channels = input_shape[0]
        self.output_shape = output_shape
        self.kernel_size = kernel_size

        self.nb_filters = nb_filters
        self.depth = depth
        self.batch_size = batch_size
        self.gap_layer = nn.MaxPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = nn.ConstantPad1d(padding=(0, 1), value=0)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.nb_filters *
                      self.depth*self.timesamples, out_features=output_shape)
        )

        modules=[]
    
        for d in range(self.depth):
            if d == 0:
                modules.append(ResBlock_PyramidalCNN(
                    in_channels=self.in_channels, out_channels=(d+1)*self.nb_filters, kernel_size=self.kernel_size,curr_depth=d))
            else:
                modules.append(ResBlock_PyramidalCNN(
                    in_channels=d*self.nb_filters, out_channels=(d+1)*self.nb_filters, kernel_size=self.kernel_size,curr_depth=d
                    ))
        self.Block = nn.Sequential(*modules)

    def forward(self, x, return_feats=False):
        tmp = self.Block(x)
        tmp = self.gap_layer_pad(tmp)
        tmp = self.gap_layer(tmp)
        tmp = tmp.view(tmp.size(0), -1)  # flatten
        output = self.output_layer(tmp)

        if return_feats:
            return tmp
        else:
            return output

