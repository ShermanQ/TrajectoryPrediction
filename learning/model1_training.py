import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.tcnn import IATCNN,nlloss
from classes.transformer import ScaledDotProduct,AttentionHead,MultiHeadAttention
from classes.tcn import TemporalConvNet
import helpers.net_training as training
import sys
import json

def get_test_tensor(test_size):
    test_tensor = torch.rand(test_size)
    lengths = torch.randint(low = 0,high = test_size[1], size = (test_size[0],1)).squeeze(1)
    lengths_ids = [torch.arange(start = n, end = test_size[1]) for n in lengths]
    active_agents = [torch.arange(start = 0, end = n) for n in lengths]

    for i,e in enumerate(lengths_ids):
        test_tensor[i,e] *= 0
    return test_tensor,active_agents

def get_nb_blocks(receptieve_field,kernel_size):
    nb_blocks = receptieve_field -1
    nb_blocks /= 2.0*(kernel_size - 1.0)
    nb_blocks += 1.0
    nb_blocks = np.log2(nb_blocks)
    nb_blocks = np.ceil(nb_blocks)

    return int(nb_blocks)

def main():
    B,Nmax,Tobs,Nfeat = 12,15,8,2
    # B,Nmax,Tobs,Nfeat = 3,5,8,2

    test_tensor,lengths_ids = get_test_tensor((B,Nmax,Tobs,Nfeat))

    num_inputs = Nfeat
    dmodel = 32
    kernel_size = 2
    dropout = 0.2
    h = 4


############# TCN #########################################
    # compute nb temporal blocks
    nb_blocks = get_nb_blocks(Tobs,kernel_size)
    num_channels = [dmodel for _ in range(nb_blocks)]
    
    # init network
    tcn = TemporalConvNet( num_inputs, num_channels, kernel_size, dropout)

    # permute channels and sequence length
    x = test_tensor.permute(0,1,3,2)  # B,Nmax,Nfeat,Tobs
    x = x.view(-1,x.size()[2],x.size()[3]) # [B*Nmax],Nfeat,Tobs


    # get ids for real agents
    # generate vector of zeros which size is the same as net output size
    # send only in the net the active agents
    # set the output values of the active agents to zeros tensor
    active_agents = torch.cat([ i*Nmax + e for i,e in enumerate(lengths_ids)],dim = 0)
    y = torch.zeros(B*Nmax,dmodel,Tobs) # [B*Nmax],Nfeat,Tobs
    y[active_agents] = tcn(x[active_agents]) # [B*Nmax],Nfeat,Tobs
 
    y = y.permute(0,2,1) # [B*Nmax],Tobs,Nfeat
    y = y.view(B,Nmax,Tobs,dmodel) # B,Nmax,Tobs,Nfeat
    conv_features = y[:,:,-1] # B,Nmax,Nfeat

############# TCN #########################################
############# TRANSFORMER #########################################

    x = conv_features
    
    # sdp = ScaledDotProduct()
    # att = sdp(x,x,x,sdp.get_mask(x,x))

    dk = dv = int(dmodel/h)
    print(dk,dv)
    # att_head = AttentionHead(dmodel,dk,dv)
    # att = att_head(x,x,x)
    # print(att.size())

    mha = MultiHeadAttention(h,dmodel,dk,dv,dropout= 0.1)
    att = mha(x,x,x)
    print(att.size())

############# TRANSFORMER #########################################



    

if __name__ == "__main__":
    main()