import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
# import torchvision
import imp
from torch.nn.utils import weight_norm
from classes.cnn import CNN
import collections

import matplotlib.pyplot as plt


def custom_mse(pred_seq,gt_seq):
    pred_seq = pred_seq.view(gt_seq.size())
    mse = nn.MSELoss(reduction= "none")

    # pred_seq = pred_seq.view(pred_seq.size()[0],12,2)
    

    # sum over a trajectory, average over batch size
    mse_loss = torch.mean(torch.sum(torch.sum(mse(pred_seq,gt_seq),dim = 2),dim = 1))

    return mse_loss

class CNN_MLP(nn.Module):
    def __init__(self,args):
        super(CNN_MLP, self).__init__()

        self.args = args

        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.input_dim = args["input_dim"]
        self.obs_len = args["input_length"]


        self.kernel_size = args["kernel_size"]
        self.coord_embedding_size = args["coord_embedding_size"]
        self.nb_conv = args["nb_conv"]
        self.nb_kernel = args["nb_kernel"]
        self.cnn_feat_size = args["cnn_feat_size"]


        self.mlp_layers = args["mlp_layers"]
        self.output_size = args["output_size"]

        self.nb_cat = args["nb_cat"]
        self.use_types = args["use_types"]
        self.word_embedding_size = args["word_embedding_size"]


        if self.coord_embedding_size > 0:
            self.coord_embedding = nn.Linear(self.input_dim,self.coord_embedding_size)
            self.cnn = CNN(num_inputs = self.coord_embedding_size,nb_kernel = self.nb_kernel,cnn_feat_size = self.cnn_feat_size,obs_len = self.obs_len,kernel_size = self.kernel_size,nb_conv = self.nb_conv)
        else:
            self.cnn = CNN(num_inputs = self.input_dim,nb_kernel = self.nb_kernel,cnn_feat_size = self.cnn_feat_size,obs_len = self.obs_len,kernel_size = self.kernel_size,nb_conv = self.nb_conv)
        # self.cnn = nn.Sequential()
        # padding = int((self.kernel_size-1)/2.0)
        # for i in range(self.nb_conv):
            
        #     conv = nn.Conv1d(self.nb_kernel , self.nb_kernel , self.kernel_size, padding=padding)

        #     if i == 0:
        #         conv = nn.Conv1d(self.coord_embedding_size, self.nb_kernel , self.kernel_size, padding=padding)
        #     self.cnn.add_module("conv0",conv)
        
        # self.project_cnn = nn.Linear(self.obs_len*self.nb_kernel,self.cnn_feat_size)
        self.mlp = nn.Sequential()

        if self.use_types == 1:
            print("types as one hot encoding")
            self.mlp.add_module("layer0",nn.Linear(self.cnn_feat_size + self.nb_cat,self.mlp_layers[0]))
        elif self.use_types == 2:
            print("types as embedding")
            self.type_embedding = nn.Embedding(self.nb_cat,self.word_embedding_size )
            self.mlp.add_module("layer0",nn.Linear(self.cnn_feat_size + self.word_embedding_size,self.mlp_layers[0]))

        else:
            print("no types")

            self.mlp.add_module("layer0",nn.Linear(self.cnn_feat_size,self.mlp_layers[0]))


        self.mlp.add_module("relu0",  nn.ReLU())
        for i in range(1,len(self.mlp_layers)):
            self.mlp.add_module("layer{}".format(i),nn.Linear(self.mlp_layers[i-1],self.mlp_layers[i]))
            self.mlp.add_module("relu{}".format(i), nn.ReLU())

        self.mlp.add_module("layer{}".format(len(self.mlp_layers)), nn.Linear(self.mlp_layers[-1],self.output_size) )

        

        

    def forward(self,x):
        types = x[1]
        x = x[0]# B,1,Obs,2  
        x = x.squeeze(1)# B,Obs,2 

        if self.coord_embedding_size > 0:
            x = self.coord_embedding(x)
            x = f.relu(x)

        # x = self.coord_embedding(x) # B,Obs,e        
        # x = f.relu(x)

        # x = x.squeeze(1)
        x = x.permute(0,2,1) # x: B,e,Tobs

        # x = self.cnn(x)# x: B,n_kernels,Tobs
        # x = x.permute(0,2,1).contiguous() # x: B,Tobs,n_kernels

        # x = x.view(self.batch_size,-1)# x: B,Tobs*n_kernels
        # x = f.relu(x) # ?

        # x = self.project_cnn(x) # x: B,cnn_feat_size

        # output = f.relu(x)

        output = self.cnn(x)
        # output = output.unsqueeze(1)

        

        
        if self.use_types == 1:        
            output = torch.cat([output,types],dim = 1)
        elif self.use_types == 2:
            types = torch.argmax(types,dim = -1)
            embedded_types = self.type_embedding(types)
            output = torch.cat([output,embedded_types],dim = 1)

        x = self.mlp(output).view(output.size()[0],int(self.output_size/self.input_dim),self.input_dim)   # x: B,tpred,2
        x = x.unsqueeze(1)   # x: B,1,tpred,2  
        return x

    