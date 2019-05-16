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

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.rnn_mlp import RNN_MLP,custom_mse
from helpers.training_class import NetTraining
# import helpers.helpers_training as training
import helpers.net_training as training
import helpers.helpers_training as helpers


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
# 10332
# 42.7 k


#python learning/rnn_mlp_training.py parameters/data.json parameters/rnn_mlp_training.json parameters/torch_extractors.json parameters/prepare_training.json
def main():
          
    # set pytorch
    torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    # torch_param = json.load(open(args[3]))
    # training_param = json.load(open(args[2]))
    # prepare_param = json.load(open(args[4]))
    training_param = json.load(open("parameters/net_training.json"))
    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))


    # open data file and load correct scene lists
    toy = prepare_param["toy"]
    data_file = torch_param["split_hdf5"]
    if prepare_param["pedestrian_only"]:
        data_file = torch_param["ped_hdf5"] 

    eval_scenes = prepare_param["eval_scenes"]

    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    if toy:
        print("toy dataset")
        data_file = torch_param["toy_hdf5"]
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
        eval_scenes = test_scenes
        train_eval_scenes = train_scenes

    scenes = [train_eval_scenes,train_scenes,test_scenes,eval_scenes]
    parameters_path = "parameters/{}.json"
    model = training_param["model"]
    
    # select model
    net_params = {}
    net = None
    if model == "rnn_mlp":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_dim" : net_params["input_dim"],
        "hidden_size" : net_params["hidden_size"],
        "recurrent_layer" : net_params["recurrent_layer"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        "nb_cat": len(prepare_param["types_dic"]),
        "use_types": net_params["use_type"],
        "word_embedding_size": net_params["word_embedding_size"],
        }
        net = RNN_MLP(args_net)

    train_loader,eval_loader = helpers.load_data_loaders(data,prepare_param,training_param,net_params,data_file,scenes)
    
    print(net)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

    args_training = {
        "n_epochs" : training_param["n_epochs"],
        "batch_size" : training_param["batch_size"],
        "device" : device,
        "train_loader" : train_loader,
        "eval_loader" : eval_loader,
        "criterion" : criterion,
        "optimizer" : optimizer,
        "use_neighbors" : net_params["use_neighbors"],
        "scalers_path" : data["scalers"],
        "plot" : training_param["plot"],
        "load_path" : net_params["load_path"],
        "plot_every" : training_param["plot_every"],
        "save_every" : training_param["save_every"],
        "offsets" : training_param["offsets"],
        "normalized" : prepare_param["normalize"],
        "net" : net,
        "print_every" : training_param["print_every"],
        "nb_grad_plots" : training_param["nb_grad_plots"],
        "nb_sample_plots" : training_param["nb_sample_plots"],
        "train" : training_param["train"]
    }

    trainer = NetTraining(args_training)
    trainer.training_loop()


    



if __name__ == "__main__":
    main()


