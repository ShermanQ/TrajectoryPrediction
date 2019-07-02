import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time
from collections import OrderedDict


from classes.fcn32s import FCN32s
# from learning.classes.fcn32s import FCN32s


class customCNN(nn.Module):
    def __init__(self,device, nb_channels_out = 512,nb_channels_projection = 128,weights_path = "./learning/data/pretrained_models/fcn32s_from_caffe.pth"):

        
        super(customCNN,self).__init__()

        self.device = device
        self.nb_channels_out = nb_channels_out
        self.weights_path = weights_path

        self.cnn = FCN32s()

        pretrained_dict = torch.load(weights_path)
        new_dict = OrderedDict()
        for k,v in pretrained_dict.items():
            if k in self.cnn.state_dict():
                new_dict[k] = v
        self.cnn.load_state_dict(new_dict)

        for param in self.cnn.parameters():
            param.requires_grad = False
                    
        self.projection = nn.Conv2d(nb_channels_out,nb_channels_projection,1)

    
        

    def forward(self,x):
        cnn_features = self.cnn(x)
        projected_features = self.projection(cnn_features)
        return projected_features

# class customCNN1(nn.Module):
#     def __init__(self,weights_path = "./learning/data/pretrained_models/fcn32s_from_caffe.pth"):

        
#         super(customCNN1,self).__init__()

#         self.weights_path = weights_path

#         self.cnn = FCN32s()

#         pretrained_dict = torch.load(weights_path)
#         new_dict = OrderedDict()
#         for k,v in pretrained_dict.items():
#             if k in self.cnn.state_dict():
#                 new_dict[k] = v
#         self.cnn.load_state_dict(new_dict)

#         for param in self.cnn.parameters():
#             param.requires_grad = False
 

#     def forward(self,x):
#         cnn_features = self.cnn(x)        
#         return cnn_features
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class customCNN1(nn.Module):
    def __init__(self,weights_path = "./learning/data/pretrained_models/fcn32s_from_caffe.pth"):

        
        super(customCNN1,self).__init__()

        

        self.cnn = torchvision.models.mobilenet_v2(pretrained=True)

        # self.cnn = torchvision.models.resnet50(pretrained=True)

        # self.cnn.avgpool = Identity()

        # self.cnn.fc = Identity()
        print(self.cnn)
        # self.cnn = torchvision.models.segmentation.fcn_resnet101(pretrained = True)  
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        
 
    
    def forward(self,x):
        cnn_features = self.cnn(x)        
        return cnn_features

class customCNN2(nn.Module):
    def __init__(self,device, nb_channels_out = 512,nb_channels_projection = 128):

        
        super(customCNN2,self).__init__()

        self.device = device
        self.nb_channels_out = nb_channels_out
        
                    
        self.projection = nn.Conv2d(nb_channels_out,nb_channels_projection,1)

    
        

    def forward(self,x):
        projected_features = self.projection(x)
        return projected_features