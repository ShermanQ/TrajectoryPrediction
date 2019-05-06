import torch
import torch.nn as nn
# import torch.nn.functional as f
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# from sklearn.model_selection import train_test_split

import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.tcnn import IATCNN,nlloss
# import helpers.helpers_training as training
import helpers.net_training as training

import sys
import json
import helpers

# import matplotlib.pyplot as plt
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
# 1498287
# 6 mo

#python learning/iatcnn_training.py parameters/data.json parameters/iatcnn_training.json parameters/torch_extractors.json parameters/prepare_training.json
def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # device = torch.device("cpu")
    # torch.backends.cudnn.benchmark = True
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
    preprocessing = json.load(open("parameters/preprocessing.json"))



    
    
    toy = prepare_param["toy"]

    nb_neighbors_max = np.array(json.load(open(torch_param["nb_neighboors_path"]))["max_neighbors"])   

    data_file = torch_param["split_hdf5"]
    train_scenes = prepare_param["train_scenes"]
    test_scenes = prepare_param["test_scenes"]


    if toy:
        print("toy dataset")
        data_file = torch_param["toy_hdf5"]
        nb_neighbors_max = np.array(json.load(open(torch_param["toy_nb_neighboors_path"]))["max_neighbors"])
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
    # else:
    #     train_scenes = helpers.helpers_training.augment_scene_list(train_scenes,preprocessing["augmentation_angles"])
    #     test_scenes = helpers.helpers_training.augment_scene_list(test_scenes,preprocessing["augmentation_angles"])
    
    
    print(nb_neighbors_max)
    train_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "train",
        use_images = False,
        data_type = "trajectories",
        use_neighbors_label = True,
        use_neighbors_sample = True,
        predict_offsets = training_param["offsets"],
        predict_smooth= training_param["predict_smooth"],
        smooth_suffix= prepare_param["smooth_suffix"]
        )

    

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "eval",
        use_images = False,
        data_type = "trajectories",
        use_neighbors_label = True,
        use_neighbors_sample = True,
        predict_offsets = training_param["offsets"],
        predict_smooth= training_param["predict_smooth"],
        smooth_suffix= prepare_param["smooth_suffix"]
        )

    # print("n_train_samples: {}".format(train_dataset.get_len()))
    # print("n_eval_samples: {}".format(eval_dataset.get_len()))


    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset)
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset)
    
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
        max_neighbors = nb_neighbors_max)

    # sum_ = 0
    # for parameter in net.parameters():
    #     sum_ += torch.flatten(parameter).size()[0]

    # print(sum_)

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
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"],
        plot_every = training_param["plot_every"], save_every = training_param["save_every"],
        offsets = training_param["offsets"], normalized = prepare_param["normalize"])



if __name__ == "__main__":
    main()



