import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.tcnn import IATCNN,nlloss
from classes.tcn import TemporalConvNet
import helpers.net_training as training
import sys
import json

def get_test_tensor(test_size):
    test_tensor = torch.rand(test_size)
    lengths_ids = [torch.arange(start = n, end = test_size[1]) for n in torch.randint(low = 0,high = test_size[1], size = (test_size[0],1)).squeeze(1)]

    for i,e in enumerate(lengths_ids):
        test_tensor[i,e] *= 0
    return test_tensor

def get_nb_blocks(receptieve_field,kernel_size):
    nb_blocks = receptieve_field -1
    nb_blocks /= 2.0*(kernel_size - 1.0)
    nb_blocks += 1.0
    nb_blocks = np.log2(nb_blocks)
    nb_blocks = np.ceil(nb_blocks)

    return int(nb_blocks)

def main():
    B,Nmax,Tobs,Nfeat = 32,48,8,2
    test_size = (B,Nmax,Tobs,Nfeat)
    test_tensor = get_test_tensor(test_size)


    
    num_inputs = Nfeat
    num_channels = []
    kernel_size = 2

    nb_blocks = get_nb_blocks(Tobs,kernel_size)
    dropout = 0.2
    tcn = TemporalConvNet( num_inputs, num_channels, kernel_size, dropout)

    

if __name__ == "__main__":
    main()