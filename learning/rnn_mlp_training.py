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

from classes.datasets import CustomDataset
from classes.rnn_mlp import RNN_MLP,custom_mse
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
#python learning/rnn_mlp_training.py parameters/data.json parameters/rnn_mlp_training.json parameters/torch_extractors.json
def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    # torch_param = json.load(open(args[3]))

    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))

    ids = np.array(json.load(open(torch_param["ids_path"]))["ids"])
    # nb_neighbors_max = np.array(json.load(open(data["prepared_ids"]))["max_neighbors"])   

    # training_param = json.load(open(args[2]))
    training_param = json.load(open("parameters/rnn_mlp_training.json"))



    # split train eval indices
    train_indices,eval_indices = train_test_split(np.array([i for i in range(training_param["nb_samples"])]),test_size = 0.2,random_state = 42)

    train_indices = np.arange(int(0.8*training_param["nb_samples"]))
    eval_indices = np.arange(int(0.8*training_param["nb_samples"]),training_param["nb_samples"])


    train_indices = ids[train_indices]
    eval_indices = ids[eval_indices]

    
    # load datasets
    train_dataset = CustomDataset(train_indices,data["torch_data"])
    eval_dataset = CustomDataset(eval_indices,data["torch_data"])

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= training_param["batch_size"], shuffle=True,num_workers= training_param["num_workers"],drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= training_param["batch_size"], shuffle=False,num_workers= training_param["num_workers"],drop_last = True)


    net = RNN_MLP(
        device = device,
        batch_size = training_param["batch_size"],
        input_dim = training_param["input_dim"],
        hidden_size = training_param["hidden_size"],
        recurrent_layer = training_param["recurrent_layer"],
        mlp_layers = training_param["mlp_layers"],
        output_size = training_param["output_size"]
    )

    print(net)
    net = net.to(device)


    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    criterion = custom_mse

    training.training_loop(training_param["n_epochs"],training_param["batch_size"],
        net,device,train_loader,eval_loader,criterion,criterion,optimizer, data["scalers"],
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"])


    load_path = "./learning/data/models/model_1552260631.156045.tar"
    checkpoint = torch.load(load_path)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)

    


    # net.eval()

    # batch_test_losses = []
    # for data in eval_loader:
    #     samples,targets = data
    #     samples = samples.to(device)
    #     targets = targets.to(device)
    #     outputs = net(samples)

if __name__ == "__main__":
    main()




