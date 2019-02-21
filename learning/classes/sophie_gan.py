import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 

"""
    Straightforward LSTM encoder

    returns hidden state at last time step
    and process last layer of hidden state through a linear layer
    to get its size to input_size
    this is a returned and will be used as starting point for decoding
"""
class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,batch_size,input_size = 2,hidden_size = 32,num_layers = 1, embedding_size = 16):
        super(encoderLSTM,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.embedding = nn.Linear(self.input_size,self.embedding_size)

        self.lstm = nn.LSTM(input_size = self.embedding_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        

        

    def forward(self,x,x_lengths,nb_max):
        self.hidden = self.init_hidden_state(self.batch_size*nb_max)

        seq_len = x.size()[1]
        print(seq_len)
        print(x.size())
        # print((self.batch_size*seq_len*nb_max,self.input_size))

        
        x = x.view(self.batch_size*seq_len*nb_max,self.input_size)
        x = self.embedding(x)
        x = x.view(self.batch_size * nb_max,seq_len,self.embedding_size)

        x = f.relu(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x,self.hidden = self.lstm(x,self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        return x[:,-1,:]



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).cuda()
        c_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).cuda()

        return (h_0,c_0)


"""
    
"""
class sophie(nn.Module):
    def __init__(self,batch_size,enc_input_size = 2,enc_hidden_size = 32,enc_num_layers = 1, embedding_size = 16):
        super(sophie,self).__init__()
        self.encoder = encoderLSTM(batch_size,enc_input_size,enc_hidden_size,enc_num_layers,embedding_size)
        self.batch_size = batch_size
        self.enc_input_size = enc_input_size
        self.enc_hidden_size = enc_hidden_size
        self.enc_num_layers = enc_num_layers
        self.embedding_size = embedding_size
        

    def forward(self,x):
        B,N,S,I = x.size()
        x = x.view(B*N,S,I)

        x_lengths,empty_trajectories_ids = self.__get_lengths_x(x.cpu().detach().numpy(),S)
        x_lengths = np.array(x_lengths)

        arg_ids = list(reversed(np.argsort(x_lengths)))
        
        x = x[arg_ids]

        sorted_x_lengths = x_lengths[arg_ids]
        x = self.encoder(x,sorted_x_lengths,N)

        rev_arg_ids = list(reversed(np.argsort(arg_ids)))
        x = x[rev_arg_ids]
        x = x.view(B,N,self.enc_hidden_size)
        print(x.size())

        
        return

    def __get_lengths_x(self,x,seq_len,padding = -1):
        x_lengths = []
        empty_trajectories_ids = []
        for i,sequence in enumerate(x):
            unpadded_length = 0
            for point in sequence:
                if point[0] != padding:
                    unpadded_length += 1
            if unpadded_length == 0:
                empty_trajectories_ids.append(i)
                unpadded_length += 1

            x_lengths.append(unpadded_length)
        return x_lengths,empty_trajectories_ids



