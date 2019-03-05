import torch
import torch.nn as nn
# import torch.nn.functional as f
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import time

from classes.datasets import CustomDataset,CustomDatasetIATCNN
from classes.tcnn import IATCNN
import helpers.helpers_training as training
import sys
import json

"""
    This script trains a iatcnn model
    The data is loaded using custom dataset
    and pytorch dataloader

    THe model is trained on 80% of the dataset and evaluated
    on the rest.
    The objective function is the mean squared error between
    target sequence and predicted sequence.

    The training evaluates as well the Final Displacement Error

    At the end of the training, model is saved in master/learning/data/models.
    If the training is interrupted, the model is saved.
    The interrupted trianing can be resumed by assigning a file path to the 
    load_path variable.

    Three curves are plotted:
        train ADE
        eval ADE
        eval FDE
"""
#python learning/iatcnn_training.py parameters/data.json parameters/iatcnn_training.json
def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    data = json.load(open("parameters/data.json"))

    ids = np.array(json.load(open(data["prepared_ids"]))["ids"])
    nb_neighbors_max = np.array(json.load(open(data["prepared_ids"]))["max_neighbors"])   

    # training_param = json.load(open(args[2]))
    training_param = json.load(open("parameters/iatcnn_training.json"))



    # split train eval indices
    train_indices,eval_indices = train_test_split(np.array([i for i in range(training_param["nb_samples"])]),test_size = 0.2,random_state = 42)

    train_indices = ids[train_indices]
    eval_indices = ids[eval_indices]

    
    # load datasets
    train_dataset = CustomDatasetIATCNN(train_indices,data["torch_data"])
    eval_dataset = CustomDatasetIATCNN(eval_indices,data["torch_data"])

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= training_param["batch_size"], shuffle=True,num_workers= training_param["num_workers"],drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= training_param["batch_size"], shuffle=False,num_workers= training_param["num_workers"],drop_last = True)

    input_dim = training_param["input_dim"]
    output_channels = training_param["output_channels"]
    net = IATCNN(
        n_inputs = input_dim * (nb_neighbors_max + 1),
        n_outputs = output_channels,
        kernel_size = training_param["kernel_size"],
        stride = training_param["stride"],
        padding = training_param["padding"])
    for data in train_loader:
        samples,labels = data
        net(samples)
    # init model and send it to gpu
    # net = 
    # net.to(device)
    
    #losses
    # criterion_train = 
    # criterion_eval = 

    #optimizer
    # optimizer = optim.Adam(net.parameters(),lr = 0.001)
    
    # train_losses,eval_losses,batch_losses = training.training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,load_path = load_path)
    

if __name__ == "__main__":
    main()




