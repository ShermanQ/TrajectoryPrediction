#F(n) = F(n-1) + [kernel_size(n)-1] * dilation(n): nth dilated causal convolution layer since input layer
# F(n) = F(n-1) + 2 * [kernel_size(n)-1] * dilation(n): nth residual causal block since input layer

# F'(n) = 1 + 2 * [kernel_size(n)-1] * (2^n -1)
# if kernel size is fixed, and the dilation of each residual block increases exponentially by 2
import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
from torch.nn.utils import weight_norm
import collections

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,n_layers, kernel_size, stride, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.net = collections.OrderedDict()
        self.conv_idx = []
        for i in range(n_layers):
            dilation = 2**i
            # padding = (i+1) * (kernel_size-1)
            padding = (kernel_size-1) * dilation 
            if i == 0:
                self.net["conv{}".format(i)] = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
            else:
                self.net["conv{}".format(i)] = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
            self.net["chomp{}".format(i)] = Chomp1d(padding)
            self.net["relu{}".format(i)] = nn.ReLU()
            self.net["dropout{}".format(i)] = nn.Dropout(dropout)
            self.conv_idx.append(i*4)

        self.net = nn.Sequential(self.net)
        self.init_weights()

    #     self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
 

    def init_weights(self):
        for index in self.conv_idx:            
            self.net[index].weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.net(x)
        return x
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
# return self.relu(out + res)


class IATCNN(nn.Module):
    def __init__(self,n_inputs, n_outputs,kernel_size, stride,  input_length,output_length,output_size,max_neighbors, dropout=0.2):
        super(IATCNN, self).__init__()

        self.right_padding = output_length - input_length
        self.n_layers = int(np.ceil(np.log2(output_length/float(kernel_size)) + 1))            
        self.n_block = int(np.ceil( (input_length + output_length)/float(2**(self.n_layers-1) * kernel_size)))
        self.net = nn.Sequential()   
        self.net.add_module("right_padding",nn.ConstantPad1d((0,self.right_padding),0.))
        for b in range(self.n_block):
            
            if b == 0:
                block = TemporalBlock(n_inputs, n_outputs,self.n_layers, kernel_size, stride, dropout=dropout)
            else:
                block = TemporalBlock(n_outputs, n_outputs,self.n_layers, kernel_size, stride, dropout=dropout)

            self.net.add_module("bloc{}".format(b),block)

        self.time_distributed = nn.Linear(n_outputs,output_size * max_neighbors)

    def forward(self,x):
        x = self.net(x).permute(0,2,1)
        x = self.time_distributed(x)
        return x
