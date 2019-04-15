import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
# import torchvision
import imp
from torch.nn.utils import weight_norm
import collections



def custom_mse(pred_seq,gt_seq):
    pred_seq = pred_seq.view(gt_seq.size())
    mse = nn.MSELoss(reduction= "none")

    # pred_seq = pred_seq.view(pred_seq.size()[0],12,2)
    

    # sum over a trajectory, average over batch size
    mse_loss = torch.mean(torch.sum(torch.sum(mse(pred_seq,gt_seq),dim = 2),dim = 1))

    return mse_loss

class RNN_MLP(nn.Module):
    def __init__(self,device,batch_size,input_dim,hidden_size,recurrent_layer,mlp_layers,output_size,nb_cat = 0,):
        super(RNN_MLP, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.hidden_size = hidden_size
        self.recurrent_layer = recurrent_layer
        self.mlp_layers = mlp_layers
        self.output_size = output_size

        self.encoder = nn.LSTM(input_size = self.input_dim,hidden_size = self.hidden_size,num_layers = self.recurrent_layer,batch_first = True)

        self.mlp = nn.Sequential()
        self.mlp.add_module("layer0",nn.Linear(self.hidden_size + nb_cat,mlp_layers[0]))
        self.mlp.add_module("relu0",  nn.ReLU())
        for i in range(1,len(mlp_layers)):
            self.mlp.add_module("layer{}".format(i),nn.Linear(self.mlp_layers[i-1],self.mlp_layers[i]))
            self.mlp.add_module("relu{}".format(i), nn.ReLU())

        self.mlp.add_module("layer{}".format(len(mlp_layers)), nn.Linear(mlp_layers[-1],self.output_size) )
        self.nb_cat = nb_cat

        

        

    def forward(self,x):
        types = x[1]
        x = x[0]
        x = x.squeeze(1)

        h = self.init_hidden_state()
        output,h = self.encoder(x,h)
        output = output[:,-1]
        if self.nb_cat > 0:
            output = torch.cat([output,types],dim = 1)
        x = self.mlp(output).view(self.batch_size,1,int(self.output_size/self.input_dim),self.input_dim)       
        return x

    def init_hidden_state(self):
        h_0 = torch.rand(self.recurrent_layer,self.batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.recurrent_layer,self.batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)