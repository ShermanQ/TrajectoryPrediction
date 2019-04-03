import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time

from classes.transformer import Transformer
from classes.tcn import TemporalConvNet

def mse_loss(outputs,targets):
    # mse = nn.MSELoss(reduction= "none")
    mse = nn.MSELoss(reduction= "mean")


    # sum over a trajectory, average over batch size
    # mse_loss = torch.mean(torch.sum(torch.sum(mse(outputs,targets),dim = 2),dim = 1))
    # mse_loss = torch.mean(torch.mean(torch.sum(mse(outputs,targets),dim = 2),dim = 1))
    mse_loss = mse(outputs,targets)

    return mse_loss

class Model1(nn.Module):
    def __init__(self,
        device,
        input_dim,
        input_length,
        kernel_size, 
        nb_blocks_transformer,
        h,
        dmodel,
        d_ff_hidden,
        dk,
        dv,
        predictor_layers,
        pred_dim,
        dropout_tcn = 0.2,
        dropout_tfr = 0.1):
        super(Model1,self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.nb_blocks_transformer = nb_blocks_transformer
        self.h = h
        self.dmodel = dmodel
        self.d_ff_hidden = d_ff_hidden
        self.dk = dk
        self.dv = dv
        self.predictor_layers = predictor_layers
        self.pred_dim = pred_dim
        self.dropout_tcn = dropout_tcn
        self.dropout_tfr = dropout_tfr

############# TCN #########################################
        # compute nb temporal blocks
        self.nb_temporal_blocks = self.__get_nb_blocks(input_length,kernel_size)        
        self.num_channels = [dmodel for _ in range(self.nb_temporal_blocks)]

        # init network
        self.tcn = TemporalConvNet(device, input_dim, self.num_channels, kernel_size, dropout_tcn)
############# TRANSFORMER #########################################

        self.encoder = Transformer(device,nb_blocks_transformer,h,dmodel,d_ff_hidden,dk,dv,dropout_tfr)

############# Predictor #########################################

        self.predictor = []
        self.predictor.append(nn.Linear(dmodel,predictor_layers[0]))
        self.predictor.append(nn.ReLU())

        for i in range(1,len(predictor_layers)):
            self.predictor.append(nn.Linear(predictor_layers[i-1], predictor_layers[i]))
            self.predictor.append(nn.ReLU())

        self.predictor.append(nn.Linear(predictor_layers[-1], pred_dim))

        self.predictor = nn.Sequential(*self.predictor)





    def forward(self,x):
        active_x = self.__get_active_ids(x)
        # permute channels and sequence length
        B,Nmax,Tobs,Nfeat = x.size()
        x = x.permute(0,1,3,2)  # B,Nmax,Nfeat,Tobs # à vérifier
        x = x.view(-1,x.size()[2],x.size()[3]) # [B*Nmax],Nfeat,Tobs

        # get ids for real agents
        # generate vector of zeros which size is the same as net output size
        # send only in the net the active agents
        # set the output values of the active agents to zeros tensor
        
        active_agents = torch.cat([ i*Nmax + e for i,e in enumerate(active_x)],dim = 0)
        y = torch.zeros(B*Nmax,self.dmodel,Tobs).to(self.device) # [B*Nmax],Nfeat,Tobs
        y[active_agents] = self.tcn(x[active_agents]) # [B*Nmax],Nfeat,Tobs

        y = y.permute(0,2,1) # [B*Nmax],Tobs,Nfeat
        y = y.view(B,Nmax,Tobs,self.dmodel) # B,Nmax,Tobs,Nfeat
        conv_features = y[:,:,-1] # B,Nmax,Nfeat

        x = conv_features
        

        y = self.encoder(x)


        y = self.predictor(y)

        t_pred = int(self.pred_dim/float(Nfeat))
        y = y.view(B,Nmax,t_pred,Nfeat) #B,Nmax,Tpred,Nfeat

        return y


    def __get_nb_blocks(self,receptieve_field,kernel_size):
        nb_blocks = receptieve_field -1
        nb_blocks /= 2.0*(kernel_size - 1.0)
        nb_blocks += 1.0
        nb_blocks = np.log2(nb_blocks)
        nb_blocks = np.ceil(nb_blocks)

        return int(nb_blocks)

    def __get_active_ids(self,x):
        nb_active = torch.sum( (torch.sum(torch.sum(x,dim = 3),dim = 2) > 0.), dim = 1).to(self.device)
        active_agents = [torch.arange(start = 0, end = n) for n in nb_active]

        return active_agents




