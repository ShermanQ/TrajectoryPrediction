import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.transformer import Encoder
from classes.model1 import Model1
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
    return test_tensor

# def get_nb_blocks(receptieve_field,kernel_size):
#     nb_blocks = receptieve_field -1
#     nb_blocks /= 2.0*(kernel_size - 1.0)
#     nb_blocks += 1.0
#     nb_blocks = np.log2(nb_blocks)
#     nb_blocks = np.ceil(nb_blocks)

#     return int(nb_blocks)

def main():
    B,Nmax,Tobs,Nfeat = 32,48,8,2
    # B,Nmax,Tobs,Nfeat = 3,5,8,2

    x = get_test_tensor((B,Nmax,Tobs,Nfeat))

  

    num_inputs = Nfeat
    dmodel = 32
    kernel_size = 2
    dropout_tcn = 0.2
    dropout_tfr = 0.1
    h = 4
    dk = dv = int(dmodel/h)
    d_ff_hidden = 4 * dmodel
    nb_blocks_transformer = 3
    print(dk,dv)


    model = Model1(num_inputs,Tobs,kernel_size,nb_blocks_transformer,h,
            dmodel,d_ff_hidden,dk,dv,dropout_tcn,dropout_tfr
    )

    y = model(x)

    print(y.size())
 



    

if __name__ == "__main__":
    main()