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

def covariance_matrix(parameters):
    mu = parameters[:2]
    sigma = parameters[2:4]
    rho = parameters[4]

    cov = rho*sigma[0]*sigma[1]
    
    cov_m = torch.diag(sigma)**2
    cov_m[1,0] = cov_m[0,1] = cov

    return mu,cov_m

def nlloss(outputs,targets,eps = 1e-15):

    outputs = outputs.contiguous().view(-1,outputs.size()[-1])
    targets = targets.contiguous().view(-1,targets.size()[-1])
    
    b,o = outputs.size()
    mu = outputs[:,:2]
    sigma1 = outputs[:,2]
    sigma2 = outputs[:,3]
    rho = outputs[:,4]
    cov = torch.mul(torch.mul(sigma1,sigma2),rho).view(b,1)
    sigma1 = (sigma1 ** 2).view(b,1)
    sigma2 = (sigma2 ** 2).view(b,1)

    matrix = torch.cat([sigma1,cov,cov,sigma2,],dim = 1).view(b,mu.size()[1],mu.size()[1])
    
    mat_inv = torch.inverse(matrix)
    diff = targets.sub(mu).view(b,mu.size()[1],1)
    a = (targets[:,0] != 0. ).cuda()
    b = (targets[:,1] != 0. ).cuda()
    c = torch.max(a,b)

    right_product = torch.bmm(mat_inv,diff)
    left_product = torch.bmm(diff.permute(0,2,1),right_product)
    
    loss = 0.5 * torch.sum(left_product[c])
    loss2 = 0.5 * torch.sum(left_product)

    return loss

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
        x = torch.tanh(x)
        return x
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
# return self.relu(out + res)


class IATCNN(nn.Module):
    def __init__(self,n_inputs, n_outputs,kernel_size, stride,  input_length,output_length,output_size,max_neighbors, dropout=0.2):
        super(IATCNN, self).__init__()

        self.right_padding = output_length - input_length
        self.max_neighbors = max_neighbors
        self.output_size = output_size
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
        # x = torch.sigmoid(x)
        b,s,_ = x.size()
        x = x.view(b,s,self.max_neighbors,self.output_size).permute(0,2,1,3)
        return x
