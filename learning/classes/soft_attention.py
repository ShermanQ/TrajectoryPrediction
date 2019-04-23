import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time


class SoftAttention(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,key_size,value_size,query_size , layers = [64,128,64]):
        super(SoftAttention,self).__init__()
    
        # self.social_attention = SoftAttention(device,batch_size,enc_hidden_size,social_features_embedding_size,dec_hidden_size ,nb_neighbors_max)

        self.device = device
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size

        self.layers = [query_size + key_size] + layers + [1]
        


        modules = []
        for i in range(1,len(self.layers)):
            modules.append(nn.Linear(self.layers[i-1],self.layers[i]))
            if i < len(self.layers) -1 :
                modules.append(nn.ReLU())
        self.core = nn.Sequential(*modules)
   

    def forward(self,q,k,v,mask = None):

        for i in range()

        hdec = hdec.unsqueeze(1)
        hdec = hdec.repeat(1,features.size()[1],1)
        features = self.features_embedding(features)
        features = f.relu(features)

        inputs = torch.cat([features,hdec],dim = 2)

     
        attn = self.core(inputs)
        attn_weigths = f.softmax(attn.permute(0,2,1), dim = 2)

       

        attn_applied = torch.bmm(attn_weigths, features)
        return attn_applied