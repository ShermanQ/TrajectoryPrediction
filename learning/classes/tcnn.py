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
import math
import numpy as np 
import time 

def nlloss(outputs,targets,eps = 1e-15):
 
    outputs = outputs.contiguous().view(-1,outputs.size()[-1])
    targets = targets.contiguous().view(-1,targets.size()[-1])
    
    b,o = outputs.size()

    mu,sigma1,sigma2,rho = outputs[:,:2],outputs[:,2],outputs[:,3],outputs[:,4]
    
    s1_s2 = torch.mul(sigma1,sigma2)
    cov = torch.mul(s1_s2,rho).view(b,1)
    sigma1, sigma2 = (sigma1 ** 2).view(b,1),(sigma2 ** 2).view(b,1)
     

    matrix = torch.cat([sigma1,cov,cov,sigma2,],dim = 1).view(b,mu.size()[1],mu.size()[1])    
    mat_inv = torch.inverse(matrix)
    # mat_inv = matrix


    diff = targets.sub(mu).view(b,mu.size()[1],1)
    

    right_product = torch.bmm(mat_inv,diff)
    square_mahalanobis = torch.bmm(diff.permute(0,2,1),right_product)
    log_num = -0.5 * square_mahalanobis.squeeze()
    
    deter = s1_s2.sub((cov.squeeze(1)) ** 2)
    epsilons = torch.ones(s1_s2.size()) * eps
    
    deter = torch.max( deter, epsilons.cuda() )
    log_deter = -0.5*  torch.log(deter)

    lloss = torch.add(log_num, log_deter)
    lloss = torch.sum(lloss)

    # log_den= -0.5 * ( 2 * math.log(2 * math.pi) + torch.log(deter) )

    # lloss =  -1.0 * torch.sum( log_num + log_den ) 
    # lloss =  -1.0 * torch.sum( log_num  ) 

    return -1.0 * lloss


# a = (targets[:,0] != 0. ).cuda()
# b = (targets[:,1] != 0. ).cuda()
# c = torch.max(a,b)
# loss = 0.5 * torch.sum(left_product[c])


class Chomp2d(nn.Module):
    def __init__(self, chomp_h_size,chomp_w_size):
        super(Chomp2d, self).__init__()
        self.chomp_h_size = chomp_h_size
        self.chomp_w_size = chomp_w_size


    def forward(self, x):
        if self.chomp_h_size != 0 and self.chomp_w_size != 0:
            return x[:, :,:-self.chomp_h_size, :-self.chomp_w_size].contiguous()

        elif self.chomp_h_size != 0 :
            return x[:, :,:-self.chomp_h_size, :].contiguous()

        elif self.chomp_w_size != 0 :
            return x[:, :,:, :-self.chomp_w_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,n_layers, kernel_size, stride, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.net = collections.OrderedDict()
        self.conv_idx = []
        for i in range(n_layers):
            dilation = (2**i,2**0)
            # dilation = (i+1,2**0)

            
            # padding = (i+1) * (kernel_size-1)
            padding_s = int(   (  (kernel_size[0]-1) * dilation[0]  )// 2  )#always divisible by 2 except for i = 0
            if i == 0:
                padding_s = padding_s + 1
            padding_t = (kernel_size[1]-1) * dilation[1]  
            
            padding = (padding_s,padding_t)

            if i == 0:
                self.net["conv{}".format(i)] = nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
                
                self.net["chomp{}".format(i)] = Chomp2d(1,padding_t)
            else:
                self.net["conv{}".format(i)] = nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
                self.net["chomp{}".format(i)] = Chomp2d(0,padding_t)     
                                       
            self.net["tanh{}".format(i)] = nn.Tanh()
            self.net["dropout{}".format(i)] = nn.Dropout(dropout)
            self.conv_idx.append(i*4)

        self.net = nn.Sequential(self.net)
        self.init_weights()


    def init_weights(self):
        for index in self.conv_idx:            
            self.net[index].weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print("a")
        x = self.net(x)
        # print("b")
        return x

class IATCNN(nn.Module):
    def __init__(self,n_inputs, n_outputs,kernel_size, stride,  input_length,output_length,output_size,max_neighbors, dropout=0.2):
        super(IATCNN, self).__init__()

        self.max_neighbors = max_neighbors
        self.output_size = output_size
        self.output_length = output_length

        n_layers_t = int(np.ceil(np.log2(input_length/float(kernel_size[1])) + 1))   
        n_layers_s = int(np.ceil(np.log2(max_neighbors/float(kernel_size[0])) + 1))     

        self.n_layers =  max(n_layers_s,n_layers_t)  

        self.time_dist_input_size = int(n_outputs//output_length)
        self.fc_layer_size = output_length * self.time_dist_input_size

        self.net = nn.Sequential()   
        
        block = TemporalBlock(n_inputs, n_outputs,self.n_layers, kernel_size, stride, dropout=dropout)
        self.net.add_module("TemporalBlock",block)

        self.fc_layer = nn.Linear(n_outputs,self.fc_layer_size)


        self.time_distributed = nn.Linear(self.time_dist_input_size,output_size)

    def forward(self,x):
        # torch.backends.cudnn.benchmark = True
        x = x.permute(0,3,1,2)
        torch.cuda.synchronize()
        s = time.time()
        x = self.net(x)

        torch.cuda.synchronize()
        print("dilated convs {}".format(time.time()-s))
        s = time.time()
        # torch.backends.cudnn.benchmark = False
        x = x[:,:,:,-1].permute(0,2,1)

        x = self.fc_layer(x)
        x = f.relu(x)

        torch.cuda.synchronize()
        print("fc layer {}".format(time.time()-s))
        s = time.time()
        b,a,_ = x.size()
        x = x.view(b,a,self.output_length,self.time_dist_input_size)
        x = self.time_distributed(x)

        torch.cuda.synchronize()
        print("time distributed {}".format(time.time()-s))
        s = time.time()

        mux_muy = x[:,:,:,:2]
        sx_sy = torch.exp(x[:,:,:,2:4])
        corr = torch.tanh(x[:,:,:,4]).unsqueeze(3)

        x = torch.cat([mux_muy,sx_sy,corr], dim = 3)

        torch.cuda.synchronize()
        print("params separation {}".format(time.time()-s))
        

        

        
        return x





# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs,n_layers, kernel_size, stride, dropout=0.2):
#         super(TemporalBlock, self).__init__()

#         self.net = collections.OrderedDict()
#         self.conv_idx = []
#         for i in range(n_layers):
#             dilation = (2**i,2**i)
#             # padding = (i+1) * (kernel_size-1)
#             padding_s = int(   (  (kernel_size[0]-1) * dilation[0]  )// 2  )#always divisible by 2 except for i = 0
#             if i == 0:
#                 padding_s = padding_s + 1
#             padding_t = (kernel_size[1]-1) * dilation[1]  
#             padding = (padding_s,padding_t)

#             if i == 0:
#                 self.net["conv{}".format(i)] = nn.Conv2d(n_inputs, n_outputs, kernel_size,
#                                             stride=stride, padding=padding, dilation=dilation)
                
#                 self.net["chomp{}".format(i)] = Chomp2d(1,padding_t)
#             else:
#                 self.net["conv{}".format(i)] = nn.Conv2d(n_outputs, n_outputs, kernel_size,
#                                             stride=stride, padding=padding, dilation=dilation)
#                 self.net["chomp{}".format(i)] = Chomp2d(0,padding_t)     
                                       
#             self.net["tanh{}".format(i)] = nn.Tanh()
#             self.net["dropout{}".format(i)] = nn.Dropout(dropout)
#             self.conv_idx.append(i*4)

#         self.net = nn.Sequential(self.net)
#         self.init_weights()


#     def init_weights(self):
#         for index in self.conv_idx:            
#             self.net[index].weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         x = self.net(x)
#         return x




# class IATCNN(nn.Module):
#     def __init__(self,n_inputs, n_outputs,kernel_size, stride,  input_length,output_length,output_size,max_neighbors, dropout=0.2):
#         super(IATCNN, self).__init__()

#         self.right_padding = output_length - input_length
#         self.max_neighbors = max_neighbors
#         self.output_size = output_size
#         self.n_layers = int(np.ceil(np.log2(output_length/float(kernel_size)) + 1))            
#         self.n_block = int(np.ceil( (input_length + output_length)/float(2**(self.n_layers-1) * kernel_size)))
#         self.net = nn.Sequential()   
#         self.net.add_module("right_padding",nn.ConstantPad1d((0,self.right_padding),0.))
#         for b in range(self.n_block):
            
#             if b == 0:
#                 block = TemporalBlock(n_inputs, n_outputs,self.n_layers, kernel_size, stride, dropout=dropout)
#             else:
#                 block = TemporalBlock(n_outputs, n_outputs,self.n_layers, kernel_size, stride, dropout=dropout)

#             self.net.add_module("bloc{}".format(b),block)

#         self.time_distributed = nn.Linear(n_outputs,output_size * max_neighbors)

#     def forward(self,x):
#         x = self.net(x).permute(0,2,1)

#         x = self.time_distributed(x)
#         # x = torch.sigmoid(x)
#         b,s,_ = x.size()
#         x = x.view(b,s,self.max_neighbors,self.output_size).permute(0,2,1,3)
#         return x

# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()


# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs,n_layers, kernel_size, stride, dropout=0.2):
#         super(TemporalBlock, self).__init__()

#         self.net = collections.OrderedDict()
#         self.conv_idx = []
#         for i in range(n_layers):
#             dilation = 2**i
#             # padding = (i+1) * (kernel_size-1)
#             padding = (kernel_size-1) * dilation 
#             if i == 0:
#                 self.net["conv{}".format(i)] = nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                             stride=stride, padding=padding, dilation=dilation)
#             else:
#                 self.net["conv{}".format(i)] = nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                             stride=stride, padding=padding, dilation=dilation)
#             self.net["chomp{}".format(i)] = Chomp1d(padding)
#             self.net["relu{}".format(i)] = nn.ReLU()
#             self.net["dropout{}".format(i)] = nn.Dropout(dropout)
#             self.conv_idx.append(i*4)

#         self.net = nn.Sequential(self.net)
#         self.init_weights()


#     def init_weights(self):
#         for index in self.conv_idx:            
#             self.net[index].weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         x = self.net(x)
#         x = torch.tanh(x)
#         return x