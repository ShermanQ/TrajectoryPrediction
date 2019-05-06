import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time


class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,input_size = 2,hidden_size = 32,num_layers = 1, embedding_size = 16):
        super(encoderLSTM,self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(self.input_size,self.embedding_size)

        self.lstm = nn.LSTM(input_size = self.embedding_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        

        

    # def forward(self,x,x_lengths,nb_max):
    def forward(self,x,x_lengths):

        hidden = self.init_hidden_state(len(x_lengths))
        x = self.embedding(x)
        x = f.relu(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x,hidden = self.lstm(x,hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        
        return x[:,-1,:], hidden[0].permute(1,2,0), hidden[1].permute(1,2,0)



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)

class Model3(nn.Module):
    def __init__(self, device, input_size = 2,output_size = 2, pred_length = 12, obs_length = 8,
                enc_hidden_size = 32, dec_hidden_size = 32, enc_num_layers = 1, dec_num_layer = 1,
                embedding_size = 16, social_features_embedding_size = 16):
        super(Model3,self).__init__()

        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.enc_num_layers = enc_num_layers
        self.dec_num_layer = dec_num_layer
        self.embedding_size = embedding_size
        self.soc_emb_size = social_features_embedding_size


        self.temp = nn.Linear(10,12)
    
    def forward(self,x):
        types = x[1]
        x = x[0] #B Nmax Tobs 2

        #list of B 1D tensors containing ids from 0 up to the last active agent
        active_x = self.__get_active_ids(x) 
        print(x.size())

        
        
        
        print(x.shape)


        

    def __get_active_ids(self,x):
        nb_active = torch.sum( (torch.sum(torch.sum(x,dim = 3),dim = 2) > 0.), dim = 1).to(self.device)
        active_agents = [torch.arange(start = 0, end = n) for n in nb_active]

        return active_agents