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

from classes.datasets import CustomDataset,Hdf5Dataset,CustomDataLoader
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

#python learning/rnn_mlp_training.py parameters/data.json parameters/rnn_mlp_training.json parameters/torch_extractors.json parameters/prepare_training.json
def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    # torch_param = json.load(open(args[3]))
    # training_param = json.load(open(args[2]))
    # prepare_param = json.load(open(args[4]))
    training_param = json.load(open("parameters/rnn_mlp_training.json"))
    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))


    ###################################################################
    
    # dset = Hdf5Dataset("parameters/data.json",
    #         "parameters/torch_extractors.json",
    #         "parameters/prepare_training.json",
    #         "train",
    #         use_images = False,
    #         data_type = "trajectories",
    #         use_neighbors = False

    #         )

    train_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= torch_param["split_hdf5"],
        scene_list= prepare_param["train_scenes"],
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "train",
        use_images = False,
        data_type = "trajectories",
        use_neighbors = False
        )

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= torch_param["split_hdf5"],
        scene_list= prepare_param["train_scenes"],
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "eval",
        use_images = False,
        data_type = "trajectories",
        use_neighbors = False
        )


    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset)
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset)
    
    

    ###################################################################

    # # ids = np.array(json.load(open(torch_param["ids_path"]))["ids"])
    # ids = json.load(open(torch_param["ids_path"]))["ids"]

    # train_ids,eval_ids,test_ids = training.split_train_eval_test(ids,prepare_param["train_scenes"],prepare_param["test_scenes"], eval_prop = prepare_param["eval_prop"])
    

    # train_indices = train_ids
    # eval_indices = test_ids

    
    # # load datasets
    # train_dataset = CustomDataset(train_indices,data["torch_data"])
    # eval_dataset = CustomDataset(eval_indices,data["torch_data"])

    # # create dataloaders
    # train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= training_param["batch_size"], shuffle=True,num_workers= training_param["num_workers"],drop_last = True)
    # eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= training_param["batch_size"], shuffle=False,num_workers= training_param["num_workers"],drop_last = True)

#############################################################################

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
        data["multiple_scalers"],training_param["model_type"],
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




