import torch
import torch.nn as nn

import random
import numpy as np 
import torchvision
import imp
import time

from classes.transformer import Transformer,MultiHeadAttention
from classes.tcn import TemporalConvNet

from classes.pretrained_vgg import customCNN

from classes.soft_attention import SoftAttention

import torch.nn.functional as f 

class SpatialAttention(nn.Module):
    def __init__(self,args):
        super(SpatialAttention,self).__init__()

        self.args = args

        self.device = args["device"]
        self.input_dim =  args["input_dim"]
        self.input_length =  args["input_length"]
        self.output_length =  args["output_length"]

        self.kernel_size =  args["kernel_size"]
        self.nb_blocks_transformer =  args["nb_blocks_transformer"]
        self.h =  args["h"]
        self.dmodel =  args["dmodel"]
        self.d_ff_hidden =  args["d_ff_hidden"]
        self.dk =  args["dk"]
        self.dv =  args["dv"]
        self.predictor_layers =  args["predictor_layers"]
        self.pred_dim =  args["pred_dim"]
        self.dropout_tcn =  args["dropout_tcn"]
        self.dropout_tfr =  args["dropout_tfr"]

        self.convnet_embedding =  args["convnet_embedding"]
        self.coordinates_embedding =  args["coordinates_embedding"]

        self.convnet_nb_layers =  args["convnet_nb_layers"]
        self.projection_layers = args["projection_layers"]
        self.spatial_projection = args["spatial_projection"]
        self.vgg_feature_size = args["vgg_feature_size"]

        self.use_tcn =  args["use_tcn"]
        self.use_mha =  args["use_mha"]


######## Dynamic part #####################################
############# x/y embedding ###############################
        self.coord_embedding = nn.Linear(self.input_dim,self.coordinates_embedding)
############# TCN #########################################
        # compute nb temporal blocks

       
        self.nb_temporal_blocks = self.__get_nb_blocks(self.input_length,self.kernel_size)        
        self.num_channels = [self.convnet_embedding for _ in range(self.nb_temporal_blocks)]

        # init network
        self.tcn = TemporalConvNet(self.device, self.coordinates_embedding, self.num_channels, self.kernel_size, self.dropout_tcn)


        # project conv features to dmodel
        self.conv2att = nn.Linear(self.convnet_embedding,self.dmodel)

        # self.conv2pred = nn.Linear(self.input_length*self.convnet_embedding,self.dmodel)
        self.conv2pred = nn.Linear(self.convnet_embedding,self.dmodel)


#########################################################

##### Spatial part ##############################################

############# features ##########################################
        self.cnn = customCNN(self.device,nb_channels_projection= self.spatial_projection)
        self.spatt2att = nn.Linear(self.spatial_projection,self.dmodel)

############# Attention #########################################
        # apply multihead attention output d_model

        if self.use_mha:
            self.mha = MultiHeadAttention(self.device,self.h,self.dmodel,self.dk,self.dv,self.dropout_tfr)
        else:
            self.mha = SoftAttention(self.device,self.dmodel,self.projection_layers,self.dropout_tfr)

################################################################
################################################################


# ############# Predictor #########################################

        # concat multihead attention and conv_features to make prediction
        self.predictor = []
        self.predictor.append(nn.Linear(self.dmodel*2,self.predictor_layers[0]))



        self.predictor.append(nn.ReLU())

        for i in range(1,len(self.predictor_layers)):
            self.predictor.append(nn.Linear(self.predictor_layers[i-1], self.predictor_layers[i]))
            self.predictor.append(nn.ReLU())

        self.predictor.append(nn.Linear(self.predictor_layers[-1], self.pred_dim))

        self.predictor = nn.Sequential(*self.predictor)





    def forward(self,x):

        # split input
        types = x[1]
        active_agents = x[2]
        points_mask = x[3][1]
        imgs = x[4]
        x = x[0]

        #### DYnamic branch ####
        # project 2d spatial coordinates into embedding space of dimension coordinates_embedding
        x = self.coord_embedding(x)
        x = torch.nn.functional.relu(x)

        # permute channels and sequence length
        B,Nmax,Tobs,Nfeat = x.size()
        x = x.permute(0,1,3,2)  # B,Nmax,Nfeat,Tobs # à vérifier
        x = x.view(-1,x.size()[2],x.size()[3]) # [B*Nmax],Nfeat,Tobs

        # filter active agents
        y = torch.zeros(B*Nmax,self.convnet_embedding,Tobs).to(self.device) # [B*Nmax],Nfeat,Tobs

        # compute sequence feature vector
        y[active_agents] = self.tcn(x[active_agents]) # [B*Nmax],Nfeat,Tobs
        y = y.permute(0,2,1) # [B*Nmax],Tobs,Nfeat
        conv_features = y.view(B,Nmax,Tobs,y.size()[2]).contiguous() # B,Nmax,Tobs,Nfeat
        # select last hidden state
        y_last = conv_features[:,:,-1]       

        # project trajectory features to be used as query in spatial attention module to dmodel dimension
        x = self.conv2att(y_last) # B,Nmax,dmodel    
        x = nn.functional.relu(x)

        # project trajectory features to be used as is, in predictor (to be concatenated to att spatial features)
        conv_features = self.conv2pred(y_last)
        conv_features = nn.functional.relu(conv_features)


        # ######################################
        ### Spatial ##############

        spatial_features = self.cnn(imgs)
        b,f,w,h = spatial_features.size()
        spatial_features = spatial_features.view(b,f,w*h).permute(0,2,1)# B,Nfeaturevectors,spatial projection
        spatial_features = self.spatt2att(spatial_features)
        spatial_features = nn.functional.relu(spatial_features) # B,Nfeaturevectors,dmodel

        ##########################
        #### Attention ###########

        q = x
        k = v = spatial_features

        spatial_att = self.mha(q,k,v)  # B,Nmax,dmodel # no need for a mask

        ############################
        #### Predictor  ############


        y = torch.cat([spatial_att,conv_features],dim = 2 ) # B,Nmax,2*dmodel

        y = self.predictor(y)

        t_pred = int(self.pred_dim/float(self.input_dim))
        # print(t_pred,self.pred_dim,self.input_dim)
        y = y.view(B,Nmax,t_pred,self.input_dim) #B,Nmax,Tpred,Nfeat

        return y


    def __get_nb_blocks(self,receptieve_field,kernel_size):
        nb_blocks = receptieve_field -1
        nb_blocks /= 2.0*(kernel_size - 1.0)
        nb_blocks += 1.0
        nb_blocks = np.log2(nb_blocks)
        nb_blocks = np.ceil(nb_blocks)

        return int(nb_blocks)

    def __get_active_ids(self,x):
        nb_active = torch.sum( (torch.sum(torch.sum(x,dim = 3),dim = 2) > 0.), dim = 1).to(self.device)
        active_agents = [torch.arange(start = 0, end = n) for n in nb_active]

        return active_agents





class ConvNet(nn.Module):
    def __init__(self,device, num_inputs, embedding,nb_layers, kernel_size, dropout=0.2,stride = 1):
        super(ConvNet, self).__init__()
        self.device = device
        layers = []

        
        for i in range(nb_layers):
            p = float( (num_inputs*(stride -1) + kernel_size - stride))/ 2.0   

            padding = (int( np.floor(p)),int( np.ceil(p)))
            layers += [ nn.ConstantPad1d(padding, 0.) ]
            
            layers += [ nn.Conv1d(embedding ,embedding,kernel_size,stride= stride)]
            
          
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x