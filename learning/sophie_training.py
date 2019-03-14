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

from classes.datasets import CustomDataset,CustomDatasetSophie
from classes.sophie_gan import sophie,discriminatorLSTM,custom_mse
import helpers.helpers_training as training
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
#
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

    ids = np.array(json.load(open(torch_param["ids_path"]))["ids"])

    train_ids,eval_ids,test_ids = training.split_train_eval_test(ids,prepare_param["train_scenes"],prepare_param["test_scenes"], eval_prop = prepare_param["eval_prop"])


    nb_neighbors_max = np.array(json.load(open(data["prepared_ids"]))["max_neighbors"])   

    
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
   
    # train_indices,eval_indices = train_test_split(np.array([i for i in range(training_param["nb_samples"])]),test_size = 0.2,random_state = 42)

    # train_indices = ids[train_indices]
    # eval_indices = ids[eval_indices]

    train_indices = train_ids
    eval_indices = eval_ids

    print(type(train_indices))


    
    # load datasets
    train_dataset = CustomDatasetSophie(train_indices,"./learning/data/")
    eval_dataset = CustomDatasetSophie(eval_indices,"./learning/data/")

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= training_param["batch_size"], shuffle=True,num_workers= training_param["num_workers"],drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= training_param["batch_size"], shuffle=False,num_workers= training_param["num_workers"],drop_last = True)

   

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
    
    generator.to(device)
    discriminator.to(device)
    
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
        training_param["plot"], training_param["load_path"])


    


if __name__ == "__main__":
    main()




