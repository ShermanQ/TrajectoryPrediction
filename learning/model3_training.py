import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.model3 import Model3

import helpers.net_training as training
import helpers
import sys
import json


#python learning/model3_training.py parameters/data.json parameters/model3_training.json parameters/torch_extractors.json parameters/prepare_training.json "parameters/preprocessing.json"

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
    training_param = json.load(open("parameters/model3_training.json"))

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
        smooth_suffix= prepare_param["smooth_suffix"],
        centers = json.load(open(data["scene_centers"])),
        augmentation = training_param["augmentation"],
        # augmentation = 0,

        augmentation_angles = training_param["augmentation_angles"]
        )

    

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "eval", ##############
        use_images = False,
        data_type = "trajectories",
        use_neighbors_label = True,
        use_neighbors_sample = True,
        predict_offsets = training_param["offsets"],
        predict_smooth= 0,
        smooth_suffix= prepare_param["smooth_suffix"],
        centers = json.load(open(data["scene_centers"])),
        augmentation = 0,
        augmentation_angles = []
        )

    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset)
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = False,dataset = eval_dataset)
    
  


    net = Model3(

        device = device, 
        input_size = training_param["input_size"],
        output_size = training_param["output_size"], 
        pred_length = training_param["pred_length"], 
        obs_length = training_param["obs_length"],
        enc_hidden_size = training_param["enc_hidden_size"], 
        dec_hidden_size = training_param["dec_hidden_size"], 
        enc_num_layers = training_param["enc_num_layers"], 
        dec_num_layer = training_param["dec_num_layer"],
        embedding_size = training_param["embedding_size"], 
        social_features_embedding_size = training_param["social_features_embedding_size"]
    )
       
    net = net.to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    # criterion = mse_loss
    criterion = nn.MSELoss(reduction= "mean")
    

    training.training_loop(training_param["n_epochs"],training_param["batch_size"],
        net,device,train_loader,eval_loader,criterion,criterion,optimizer,data["scalers"],
        data["multiple_scalers"],training_param["model_type"],
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"],
        plot_every = training_param["plot_every"],offsets = training_param["offsets"], save_every = training_param["save_every"])

    


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
