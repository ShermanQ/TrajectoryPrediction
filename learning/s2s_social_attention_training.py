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
from classes.s2s_social_att import S2sSocialAtt
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
#python learning/s2s_social_attention_training.py parameters/data.json parameters/s2s_social_attention_training.json parameters/torch_extractors.json parameters/prepare_training.json
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
    training_param = json.load(open("parameters/s2s_social_attention_training.json"))
    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))
    preprocessing = json.load(open("parameters/preprocessing.json"))



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

        
    train_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_eval_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "train_eval", # train
        use_images = False,
        data_type = "trajectories",
        use_neighbors = True,
        use_masks = 1,
        predict_offsets = training_param["offsets"],
        predict_smooth= training_param["predict_smooth"],
        smooth_suffix= prepare_param["smooth_suffix"],
        centers = json.load(open(data["scene_centers"])),
        padding = prepare_param["padding"],

        augmentation = 0,
        augmentation_angles = training_param["augmentation_angles"],
        normalize =prepare_param["normalize"]
        )

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= test_scenes, #eval_scenes
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "test", #eval
        use_images = False,
        data_type = "trajectories",
        use_neighbors = True,
        use_masks = 1,
        predict_offsets = training_param["offsets"],
        predict_smooth= 0,
        smooth_suffix= prepare_param["smooth_suffix"],
        centers = json.load(open(data["scene_centers"])),
        padding = prepare_param["padding"],

        augmentation = 0,
        augmentation_angles = [],
        normalize =prepare_param["normalize"]


        )

    # print("n_train_samples: {}".format(train_dataset.get_len()))
    # print("n_eval_samples: {}".format(eval_dataset.get_len()))

    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset,test=training_param["test"])
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset,test=training_param["test"])
    

    args = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_dim" : training_param["input_dim"],
        "enc_hidden_size" : training_param["enc_hidden_size"],
        "enc_num_layers" : training_param["enc_num_layers"],
        "dec_hidden_size" : training_param["dec_hidden_size"],
        "dec_num_layer" : training_param["dec_num_layer"],

        "embedding_size" : training_param["embedding_size"],
        "output_size" : training_param["output_size"],
        "pred_length" : training_param["pred_length"],
        "projection_layers" : training_param["projection_layers"],
        "enc_feat_embedding" : training_param["enc_feat_embedding"],
        "condition_decoder_on_outputs" : training_param["condition_decoder_on_outputs"]


    }

    net = S2sSocialAtt(args)

    # sum_ = 0
    # for parameter in net.parameters():
    #     sum_ += torch.flatten(parameter).size()[0]

    # print(sum_)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)

    print(net)
    net = net.to(device)


    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    # criterion = custom_mse
    # criterion = nn.MSELoss(reduction= "mean")
    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))



    training.training_loop(training_param["n_epochs"],training_param["batch_size"],
        net,device,train_loader,eval_loader,criterion,criterion,optimizer, data["scalers"],
        data["multiple_scalers"],training_param["model_type"],
        plot = training_param["plot"],early_stopping = True,load_path = training_param["load_path"],
        plot_every = training_param["plot_every"], save_every = training_param["save_every"],
        offsets = training_param["offsets"], normalized = prepare_param["normalize"])


    # load_path = "./learning/data/models/model_1552260631.156045.tar"
    # checkpoint = torch.load(load_path)
    # net.load_state_dict(checkpoint['state_dict'])
    # net = net.to(device)

    


    # net.eval()

    # batch_test_losses = []
    # for data in eval_loader:
    #     samples,targets = data
    #     samples = samples.to(device)
    #     targets = targets.to(device)
    #     outputs = net(samples)

if __name__ == "__main__":
    main()




