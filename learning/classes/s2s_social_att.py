import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import numpy as np
import torchvision
import imp
import time

# class decoderLSTM(nn.Module):
#     # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

#     def __init__(self,device,batch_size,input_size,output_size,hidden_size,num_layers):
#         super(decoderLSTM,self).__init__()

#         self.device = device

#         self.batch_size = batch_size
#         self.input_size = input_size

#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers


#         self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_size,num_layers = num_layers,batch_first = True)
#         self.out = nn.Linear(self.hidden_size,self.output_size)



#     def forward(self,x,hidden):
#         # self.hidden = self.init_hidden_state()
#         # x = x.view(self.seq_len,self.batch_size,2)
#         output,self.hidden = self.lstm(x,hidden)
#         output = self.out(output)#activation?
#         return output, hidden

#     def init_hidden_state(self):
#         h_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)
#         c_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)

#         return (h_0,c_0)

class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,device,batch_size,input_size,hidden_size,num_layers,embedding_size):
        super(encoderLSTM,self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size


        self.lstm = nn.LSTM(input_size = self.embedding_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)


    # def forward(self,x,x_lengths,nb_max):
    def forward(self,x,x_lengths):

        hidden = self.init_hidden_state(len(x_lengths))
        

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x,hidden = self.lstm(x,hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        # hidden tuple( B*N enc,B*N enc)
        # B*N enc

        return x[:,-1,:], hidden[0][0], hidden[1][0]



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)



class S2sSocialAtt(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,args):
        super(S2sSocialAtt,self).__init__()

        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.input_dim = args["input_dim"]
        self.enc_hidden_size = args["enc_hidden_size"]
        self.enc_num_layers = args["enc_num_layers"]
        self.dec_hidden_size = args["dec_hidden_size"]
        self.dec_num_layer = args["dec_num_layer"]
        self.embedding_size = args["embedding_size"]
        self.output_size = args["output_size"]
        self.pred_length = args["pred_length"]

        self.coordinates_embedding = nn.Linear(self.input_dim,self.embedding_size)
        self.coordinates_reverse_embedding = nn.Linear(self.embedding_size,self.output_size)

        self.encoder = encoderLSTM(self.device,self.batch_size,self.input_dim,self.enc_hidden_size,self.enc_num_layers,self.embedding_size)


    def forward(self,x):
        types = x[1]
        active_agents = x[2]
        points_mask = x[3][1]
        points_mask_in = x[3][0]

        imgs = x[4]

        x = x[0] # B N S 2

        # embed coordinates
        x_e = self.coordinates_embedding(x) # B N S E
        x_e = f.relu(x_e)
        B,N,S,E = x_e.size()

        x_e = x_e.view(B*N,S,E)# B*N S E

        # get lengths for padding
        x_lengths = np.sum(points_mask_in[:,:,:,0],axis = -1).reshape(B*N)
        x_lengths = np.add(x_lengths, (x_lengths == 0).astype(int)) # put length 1 to padding agent, not efficient but practical
        
        # get the indices of the descending sorted lengths
        arg_ids = list(reversed(np.argsort(x_lengths)))

        # order input vector based on descending sequence lengths
        x_e = x_e[arg_ids]
        # get ordered/unpadded sequences lengths for pack_padded object   
        sorted_x_lengths = x_lengths[arg_ids] 
        encoder_hiddens,h,c = self.encoder(x_e,sorted_x_lengths)

        # reverse ordering of indices
        rev_arg_ids = np.argsort(arg_ids)
        # reverse ordering of encoded sequence
        encoder_hiddens = encoder_hiddens[rev_arg_ids]
        h = h[rev_arg_ids]
        c = c[rev_arg_ids]

        # reshape to original batch_size
        encoder_hiddens = encoder_hiddens.view(B,N,self.enc_hidden_size)
        h = h.view(B,N,self.enc_hidden_size)
        c = c.view(B,N,self.enc_hidden_size)





        print("a")




