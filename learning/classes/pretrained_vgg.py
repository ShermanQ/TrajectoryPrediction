import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time

class customCNN(nn.Module):
    def __init__(self,device,nb_channels_in = 3, nb_channels_out = 512,nb_channels_projection = 128, input_size = 224, output_size = 7, embedding_size = 16,weights_path = "./learning/data/pretrained_models/voc_fc32_state.tar"):

        
        super(customCNN,self).__init__()

        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.embedding_size = embedding_size
        self.nb_channels_in = nb_channels_in
        self.nb_channels_out = nb_channels_out
        self.weights_path = weights_path

        # out number of features is 25088 = 512 * 7 * 7 
        # self.cnn = torchvision.models.vgg19(pretrained=True).features
        # self.__init_cnn()

        self.cnn = torchvision.models.vgg16(pretrained=False).features
        # print(self.cnn)
        
        self.cnn.load_state_dict(torch.load(self.weights_path)["state_dict"])
        for param in self.cnn.parameters():
            param.requires_grad = False
        # main_model = imp.load_source('MainModel', "./learning/data/pretrained_models/vgg16_voc.py")
        # self.cnn = torch.load("./learning/data/pretrained_models/vgg16_voc.pth").to(device)
        # print(self.cnn)

        
        # self.embedding = nn.Linear(self.input_size,self.embedding_size)
        self.projection = nn.Conv2d(nb_channels_out,nb_channels_projection,1)
        self.nb_channels_projection = nb_channels_projection
        # self.embedding = nn.Linear(output_size**2, embedding_size)

        
    # def __init_cnn(self):
    #     # print(torchvision.models.vgg16(pretrained=False))
    #     self.cnn = torchvision.models.vgg16(pretrained=False).features
    #     # print(self.cnn)
        
    #     self.cnn.load_state_dict(torch.load(self.weights_path)["state_dict"])
    #     for param in self.cnn.parameters():
    #         param.requires_grad = False
    #     # self.cnn = self.cnn.to(self.device)

        

    def forward(self,x):
        # x = x.view(self.batch_size,self.nb_channels_in,self.input_size,self.input_size)
        # torch.cuda.synchronize()
        cnn_features = self.cnn(x)
        

        projected_features = self.projection(cnn_features)
        

        # print("test {}".format(cnn_features.size()))

        # cnn_features = cnn_features.view(self.batch_size,-1)
        # projected_features = projected_features.view(self.batch_size,self.nb_channels_projection,self.output_size**2) 
        # projected_features = projected_features.permute(0,2,1)
        # projected_features = self.projection(cnn_features).view(self.batch_size,self.output_size**2) # B * 49

        # embedded_features = self.embedding(projected_features).view(self.batch_size,self.embedding_size)

        # return embedded_features
        return projected_features