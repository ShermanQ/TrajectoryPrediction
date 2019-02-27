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
from classes.sophie_gan import sophie,discriminatorLSTM,custom_mse
import helpers.helpers_training as training
import torchvision.models as models
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

def main():
          
    # set pytorch
    # torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    # parameters
    
    
    batch_size= 32 #20
    obs_length= 8
    pred_length  = 12

    enc_input_size = 2
    enc_hidden_size = 32
    enc_num_layers = 1
    embedding_size = 16

    disc_hidden_size = 64
    disc_nb_layer = 1
    output_size = 2

    nb_samples = 1000
    num_workers = 0

    lr_gen = 0.001
    lr_disc = 0.001

    n_epochs = 2

    load_path = None
    # load_path = "./learning/data/models/sophie_1551128311.430322.tar"
    # split train eval indices
    train_indices,eval_indices = train_test_split(np.array([i for i in range(nb_samples)]),test_size = 0.2,random_state = 42)

    print(type(train_indices))


    
    # load datasets
    train_dataset = CustomDatasetIATCNN(train_indices,"./learning/data/")
    eval_dataset = CustomDatasetIATCNN(eval_indices,"./learning/data/")

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= batch_size, shuffle=True,num_workers= num_workers,drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= batch_size, shuffle=False,num_workers= num_workers,drop_last = True)

   

    # init model and send it to gpu
    generator = sophie(device,batch_size,
                enc_input_size,
                enc_hidden_size,
                enc_num_layers, 
                embedding_size)

    discriminator = discriminatorLSTM(
        device,
        batch_size,
        enc_input_size,
        embedding_size,
        disc_hidden_size,
        disc_nb_layer,
        pred_length + obs_length)
    
    generator.to(device)
    discriminator.to(device)
    
    #losses
    criterion_gan = nn.BCELoss()
    criterion_gen = custom_mse
    # criterion_train = 
    # criterion_eval = 

    # optimizer
    optimizer_gen = optim.Adam(generator.parameters(),lr = lr_gen)
    optimizer_disc = optim.Adam(discriminator.parameters(),lr = lr_disc)

    
    # train_losses,eval_losses,batch_losses = training.training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,load_path = load_path)
    
    # for batch_idx, data in enumerate(train_loader):
    training.sophie_training_loop(n_epochs,batch_size,generator,discriminator,optimizer_gen,optimizer_disc,device,
        train_loader,eval_loader,obs_length, criterion_gan,criterion_gen, 
        pred_length, output_size,True, load_path)


    


if __name__ == "__main__":
    main()




