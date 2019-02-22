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

        # number of active agent per batch
        agent_numbers = self.__get_number_agents(x)
        
        # reshape so that new batch size is former batch_size * nb_max agents
        output = x.view(B*N,S,I)

        # get lengths of sequences(without padding) in ascending order
        x_lengths = self.__get_lengths_x(output.cpu().detach().numpy(),S)
        x_lengths = np.array(x_lengths)

        
        
        

        # get the indices of the descending sorted lengths
        arg_ids = list(reversed(np.argsort(x_lengths)))

        # order input vector based on descending sequence lengths
        output = output[arg_ids]

        # get ordered/unpadded sequences lengths for pack_padded object
        sorted_x_lengths = x_lengths[arg_ids]

        # encode
        output = self.encoder(output,sorted_x_lengths,N)

        # reverse ordering of indices
        rev_arg_ids = np.argsort(arg_ids)
        # reverse ordering of encoded sequence
        output = output[rev_arg_ids]
        # reshape to original batch_size
        output = output.view(B,N,self.enc_hidden_size)

        
      
        # get sequences last points
        last_points = torch.stack([s[i-1,:] for i,s in zip(x_lengths,x.view(B*N,S,I))])
        last_points = last_points.view(B,N,I)

        # get main agent last point
        reference_points = last_points[:,0].view(B,1,self.enc_input_size).repeat(1,N,1)
       
        # compute euclidean distance
        dist = torch.sqrt(torch.sum(torch.pow(last_points-reference_points,2),dim = 2))

        print(dist)

        print(dist.size())
    

        
        return

    def __get_lengths_x(self,x,seq_len,padding = -1):
        x_lengths = []
        for i,sequence in enumerate(x):
            unpadded_length = 0
            for point in sequence:
                if point[0] != padding:
                    unpadded_length += 1
            if unpadded_length == 0:
            #     empty_trajectories_ids.append(i)
                unpadded_length += 1

            x_lengths.append(unpadded_length)
        return x_lengths

    def __get_number_agents(self,x,padding = -1):
        agent_numbers = []
        for sample in x:
            nb_agent = 0
            for sequence in sample:
                # if sequence[-1][0] != padding:
                if sequence[0][0] != padding:                
                    nb_agent += 1
            agent_numbers.append(nb_agent)
        return agent_numbers






