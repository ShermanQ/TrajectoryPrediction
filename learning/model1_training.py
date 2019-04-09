import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.model1 import Model1,mse_loss
import helpers.net_training as training
import helpers
import sys
import json


#python learning/model1_training.py parameters/data.json parameters/model1_training.json parameters/torch_extractors.json parameters/prepare_training.json "parameters/preprocessing.json"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # device = torch.device("cpu")
    print(device)
    print(torch.cuda.is_available())
    
        
    args = sys.argv    

    # data = json.load(open(args[1]))
    # training_param = json.load(open(args[2]))
    # torch_param = json.load(open(args[3]))
    # prepare_param = json.load(open(args[4]))
    # preprocessing = json.load(open(args[5]))


    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))
    training_param = json.load(open("parameters/model1_training.json"))
    preprocessing = json.load(open("parameters/preprocessing.json"))



    toy = prepare_param["toy"]
    nb_neighbors_max = np.array(json.load(open(torch_param["nb_neighboors_path"]))["max_neighbors"])   


    data_file = torch_param["split_hdf5"]
    train_scenes = prepare_param["train_scenes"]
    test_scenes = prepare_param["test_scenes"]

    if toy:
        print("toy dataset")
        data_file = torch_param["toy_hdf5"]
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
        nb_neighbors_max = np.array(json.load(open(torch_param["toy_nb_neighboors_path"]))["max_neighbors"])
    else:
        train_scenes = helpers.helpers_training.augment_scene_list(train_scenes,preprocessing["augmentation_angles"])
        test_scenes = helpers.helpers_training.augment_scene_list(test_scenes,preprocessing["augmentation_angles"])

    

    print(nb_neighbors_max)
    train_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "train",
        use_images = False,
        data_type = "frames",
        use_neighbors_label = True,
        use_neighbors_sample = True
        )

    

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "eval", ##############
        use_images = False,
        data_type = "frames",
        use_neighbors_label = True,
        use_neighbors_sample = True
        )

    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset,test= training_param["test"])
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset,test= training_param["test"])
    
  


    net = Model1(
        device = device,
        input_dim = training_param["input_dim"],
        input_length = training_param["obs_length"],
        output_length = training_param["pred_length"],
        kernel_size = training_param["kernel_size"],
        nb_blocks_transformer = training_param["nb_blocks"],
        h = training_param["h"],
        dmodel = training_param["dmodel"],
        d_ff_hidden = 4 * training_param["dmodel"],
        dk = int(training_param["dmodel"]/training_param["h"]),
        dv = int(training_param["dmodel"]/training_param["h"]),
        predictor_layers = training_param["predictor_layers"],
        pred_dim = training_param["pred_length"] * training_param["input_dim"] ,
        dropout_tcn = training_param["dropout_tcn"],
        dropout_tfr = training_param["dropout_tfr"]
    )

    net = net.to(device)
    # print(net)

    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    criterion = mse_loss
    

    training.training_loop(training_param["n_epochs"],training_param["batch_size"],
        net,device,train_loader,eval_loader,criterion,criterion,optimizer,data["scalers"],
        data["multiple_scalers"],training_param["model_type"],
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"],
        plot_every = training_param["plot_every"], save_every = training_param["save_every"])

if __name__ == "__main__":
    main()







# def get_test_tensor(test_size):
#     test_tensor = torch.rand(test_size)
#     lengths = torch.randint(low = 0,high = test_size[1], size = (test_size[0],1)).squeeze(1)
#     lengths_ids = [torch.arange(start = n, end = test_size[1]) for n in lengths]
#     active_agents = [torch.arange(start = 0, end = n) for n in lengths]

#     for i,e in enumerate(lengths_ids):
#         test_tensor[i,e] *= 0
#     return test_tensor
# x = get_test_tensor((B,Nmax,Tobs,Nfeat))
# def get_nb_blocks(receptieve_field,kernel_size):
#     nb_blocks = receptieve_field -1
#     nb_blocks /= 2.0*(kernel_size - 1.0)
#     nb_blocks += 1.0
#     nb_blocks = np.log2(nb_blocks)
#     nb_blocks = np.ceil(nb_blocks)

#     return int(nb_blocks)
