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
from classes.sophie_gan import sophie,discriminatorLSTM,custom_mse
# import helpers.helpers_training as training
import helpers.gan_training as training
import helpers


import torchvision.models as models

import json 
import sys
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
# 14796980
# 14716001
# 80979
# 59,2 Mo
# python learning/sophie_training.py parameters/data.json parameters/sophie_training.json parameters/torch_extractors.json  parameters/prepare_training.json
def main():
    args = sys.argv   

    # data = json.load(open(args[1]))
    #training_param = json.load(open(args[2]))
    # torch_param = json.load(open(args[3]))
    # prepare_param = json.load(open(args[4]))

    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    training_param = json.load(open("parameters/sophie_training.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))

      
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    device = torch.device("cpu")  
    print(device)
    print(torch.cuda.is_available())
        
   
    
    nb_neighbors_max = np.array(json.load(open(torch_param["nb_neighboors_path"]))["max_neighbors"]) - 1  


    toy = prepare_param["toy"]
 

    data_file = torch_param["split_hdf5"]
    train_scenes = prepare_param["train_scenes"]
    test_scenes = prepare_param["test_scenes"]


    if toy:
        print("toy dataset")
        data_file = torch_param["toy_hdf5"]
        nb_neighbors_max = np.array(json.load(open(torch_param["toy_nb_neighboors_path"]))["max_neighbors"])  - 1
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"]
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
        use_images = True,
        data_type = "trajectories",
        use_neighbors_label = True,
        use_neighbors_sample = True
        )

    eval_dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = "eval",
        use_images = True,
        data_type = "trajectories",
        use_neighbors_label = True,
        use_neighbors_sample = True
        )

    # print("n_train_samples: {}".format(train_dataset.get_len()))
    # print("n_eval_samples: {}".format(eval_dataset.get_len()))

    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset)
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset)
    

    # init model and send it to gpu
    generator = sophie(device,
                batch_size = training_param["batch_size"],
                enc_input_size = training_param["encoder_input_size"],
                enc_hidden_size = training_param["encoder_hidden_size"],
                enc_num_layers = training_param["encoder_num_layers"],
                embedding_size = training_param["embedding_size"],
                dec_hidden_size = training_param["decoder_hidden_size"],
                nb_neighbors_max = nb_neighbors_max,
                social_features_embedding_size = training_param["social_features_embedding_size"],
                gaussian_dim = training_param["gaussian_dimension"],
                dec_num_layer = training_param["decoder_num_layer"],
                output_size = training_param["output_size"],
                pred_length = training_param["pred_length"],
                obs_length = training_param["obs_length"]
    )

    
    discriminator = discriminatorLSTM(
        device,
        batch_size = training_param["batch_size"],
        input_size = training_param["encoder_input_size"],
        embedding_size = training_param["embedding_size"],
        hidden_size = training_param["discriminator_hidden_size"],
        num_layers = training_param["discriminator_nb_layer"],
        seq_len = training_param["seq_length"]
    )
    
    print(generator)
    print(discriminator)

    # sum_ = 0
    # # for parameter in generator.parameters():
    # #     sum_ += torch.flatten(parameter).size()[0]

    # for parameter in generator.cnn.parameters():
    #     sum_ += torch.flatten(parameter).size()[0]
    # print(sum_)

    generator.to(device)
    torch.cuda.synchronize()
    discriminator.to(device)
    torch.cuda.synchronize()

    
    #losses
    criterion_gan = nn.BCELoss()
    criterion_gen = custom_mse

    # optimizer
    optimizer_gen = optim.Adam(generator.parameters(),lr = training_param["lr_generator"])
    optimizer_disc = optim.Adam(discriminator.parameters(),lr = training_param["lr_discriminator"])

        
    # for batch_idx, data in enumerate(train_loader):
    training.sophie_training_loop(training_param["n_epochs"],training_param["batch_size"],generator,discriminator,optimizer_gen,optimizer_disc,device,
        train_loader,eval_loader,training_param["obs_length"], criterion_gan,criterion_gen, 
        training_param["pred_length"], training_param["output_size"],data["scalers"],data["multiple_scalers"],
        training_param["plot"], training_param["load_path"],plot_every = training_param["plot_every"], save_every = training_param["save_every"])


    


if __name__ == "__main__":
    main()




