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
from classes.tcnn import IATCNN,nlloss
import helpers.helpers_training as training
import sys
import json
import matplotlib.pyplot as plt
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
#python learning/iatcnn_training.py parameters/data.json parameters/iatcnn_training.json parameters/torch_extractors.json parameters/prepare_training.json
def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    torch.backends.cudnn.benchmark = True
    # device = "cpu"
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    # training_param = json.load(open(args[2]))
    # torch_param = json.load(open(args[3]))
    # prepare_param = json.load(open(args[4]))

    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))
    training_param = json.load(open("parameters/iatcnn_training.json"))

    ids = np.array(json.load(open(torch_param["ids_path"]))["ids"])

    nb_neighbors_max = np.array(json.load(open(data["prepared_ids"]))["max_neighbors"])   

    
    



    # split train eval indices
    train_ids,eval_ids,test_ids = training.split_train_eval_test(ids,prepare_param["train_scenes"],prepare_param["test_scenes"], eval_prop = prepare_param["eval_prop"])

    train_indices = train_ids
    eval_indices = eval_ids
   

    
    # load datasets
    train_dataset = CustomDatasetIATCNN(train_indices,data["torch_data"])
    eval_dataset = CustomDatasetIATCNN(eval_indices,data["torch_data"])

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= training_param["batch_size"], shuffle=True,num_workers= training_param["num_workers"],drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= training_param["batch_size"], shuffle=False,num_workers= training_param["num_workers"],drop_last = True)

   
    input_dim = training_param["input_dim"]
    output_channels = training_param["output_channels"]
    net = IATCNN(
        n_inputs = input_dim,
        n_outputs = output_channels,
        kernel_size = training_param["kernel_size"],
        stride = training_param["stride"],
        input_length = training_param["obs_length"],
        output_length = training_param["pred_length"],
        output_size = training_param["output_size"],
        max_neighbors = nb_neighbors_max + 1)



    # load_path = "./learning/data/models/model_1552166089.4612148.tar"
    # checkpoint = torch.load(load_path)
    # net.load_state_dict(checkpoint['state_dict'])
    
    net = net.to(device)
    print(net)
    

    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    criterion = nlloss

    training.training_loop(training_param["n_epochs"],training_param["batch_size"],
        net,device,train_loader,eval_loader,criterion,criterion,optimizer,data["scalers"],
        data["multiple_scalers"],training_param["model_type"],
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"])



if __name__ == "__main__":
    main()




