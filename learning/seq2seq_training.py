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
from classes.seq2seq_model import seq2seq
import helpers.helpers_training as training

def main():
    # data_path = "./data/deep/data.csv"
    # label_path = "./data/deep/labels.csv"

    # samples_path = "./learning/data/samples/"
    # labels_path = "./learning/data/labels/"
    
    # extract_tensors(data_path,label_path,samples_path,labels_path)
    # print(time.time()-s)
       
    # set pytorch
    torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    # parameters
    input_size,hidden_size,num_layers,output_size,batch_size,seq_len,seq_len_d  = 2,30,3,2,200,8,12

    nb_samples = 42380
    learning_rate = 0.001
    n_epochs = 10

    load_path = None
    # load_path = "./learning/data/models/model_1550002069.1353667.tar"

    # split train eval indices
    train_indices,eval_indices = train_test_split([i for i in range(nb_samples)],test_size = 0.2,random_state = 42)

    # load datasets
    train_dataset = CustomDataset(train_indices,"./learning/data/")
    eval_dataset = CustomDataset(eval_indices,"./learning/data/")

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= batch_size, shuffle=True,num_workers= 10,drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= batch_size, shuffle=False,num_workers= 10,drop_last = True)

    # init model and send it to gpu
    net = seq2seq(input_size,output_size,hidden_size,hidden_size,num_layers,num_layers,batch_size,seq_len_d)
    net.to(device)
    
    #losses
    criterion_train = nn.MSELoss()
    criterion_eval = nn.MSELoss(reduction="sum")
    criterion_eval = criterion_train

    #optimizer
    optimizer = optim.Adam(net.parameters(),lr = 0.001)
    
    train_losses,eval_losses,batch_losses = training.training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,load_path = load_path)
    

if __name__ == "__main__":
    main()




