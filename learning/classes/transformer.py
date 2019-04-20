import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time

"""
    [[0, 0, 0, 0, 1, 1, 1, 1,1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    mask shape for one batch element
    the one will be set to -inf so softmax doesnt count them while normalizing
    the full 0 rows correspond to padding agent/padding agent 
    dot products
    the 1 correspond to real agent/padding agent dot products
    the 0 on non full 0 rows correspond to real agent/real agent  dot products
"""
class ScaledDotProduct(nn.Module):
    def __init__(self,device,dropout = 0.1):
        super(ScaledDotProduct, self).__init__()
        self.device = device
    def forward(self,q,k,v,mask = None):
        # q: B*Sq*e
        # k: B*Sk*e
        # v: B*Sv*e
        # in practice sq = sk = sv

        scale_factor = np.sqrt(q.size()[-1])

        min_inf = float('-inf')

        # dot product
        att = torch.bmm(q, torch.transpose(k,2,1)) # B * Sq *Sk

        # scale 
        att /= scale_factor

        # mask
        if mask is not None:
            att = att.masked_fill(mask,min_inf)
        
        # softmax
        att_weigths = f.softmax(att,dim = 2 ) # B * Sq *Sk

        


        # matmul

        att = torch.bmm(att_weigths,v)

        return att

    def get_mask(self,points_mask,max_batch):
        # compute mask put one where the dot product conv_features*conv_features.T is 0.
        # and the sum over a given row of the resulting matrice is gt 1
        # Nmax = q.size()[1]
        # dot = torch.bmm(q, torch.transpose(k,2,1))
        # mask = (dot == 0) & (torch.sum(dot,dim = 1) > 0.).unsqueeze(2).repeat(1,1,Nmax)
        # mask = mask.to(self.device)
        # return mask

        sample_sum = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) > 0).astype(int)
        a = np.repeat(np.expand_dims(sample_sum,axis = 2),max_batch,axis = -1)
        b = np.transpose(a,axes=(0,2,1))
        mha_mask = np.logical_and(np.logical_xor(a,b),a).astype(int)
        return torch.ByteTensor(mha_mask).to(self.device)
        

class AttentionHead(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,dk,dv,dropout = 0.1):
        super(AttentionHead,self).__init__()
        self.device = device
        self.q_projection = nn.Linear(dmodel,dk)
        self.k_projection = nn.Linear(dmodel,dk)
        self.v_projection = nn.Linear(dmodel,dv)
        self.dot_attention = ScaledDotProduct(device,dropout)

    def forward(self,q,k,v,points_mask):
        Q = self.q_projection(q)
        K = self.k_projection(k)
        V = self.v_projection(v)
        att = self.dot_attention(Q,K,V,self.dot_attention.get_mask(points_mask,Q.size()[1]))
        return att #B,Nmax,dv

class MultiHeadAttention(nn.Module):
    def __init__(self,device,h,dmodel,dk,dv,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        assert dmodel == h*dv
        assert dk == dv
        self.device = device
        self.heads  = nn.ModuleList()
        for i in range(h):
            self.heads.append(AttentionHead(device,dmodel,dk,dv,dropout))  # inefficient#############################
        

        self.multihead_projection = nn.Linear(h*dv,dmodel)

    def forward(self,q,k,v,points_mask ):

        atts = [] #H,Nmax,dv
        for head in self.heads:
            atts.append(head(q,k,v,points_mask))#B,Nmax,dv

        atts = torch.cat(atts,dim = 2) #B,Nmax,dv * h
        out = self.multihead_projection(atts) #B,Nmax,dmodel

        return out

        
        

class EncoderBlock(nn.Module):
    def __init__(self,device,h,dmodel,d_ff_hidden,dk,dv,dropout = 0.1):
        super(EncoderBlock,self).__init__()
        self.multihead_att = MultiHeadAttention(device,h,dmodel,dk,dv,dropout )


        self.device = device
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(dmodel,d_ff_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_ff_hidden,dmodel)
        # )
        self.dropout = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(dmodel)
        # self.norm_layer2 = nn.LayerNorm(dmodel)


    def forward(self,x,mask):
        # x = self.norm_layer1( x + self.dropout(self.multihead_att(x,x,x,mask)) ) #B,Nmax,dmodel
        x = self.norm_layer1( self.dropout(self.multihead_att(x,x,x,mask)) ) #B,Nmax,dmodel

        x = self.dropout(self.multihead_att(x,x,x,mask))  #B,Nmax,dmodel


        
        # x = self.norm_layer2( x + self.dropout( self.feed_forward(x) ) )

        # x =  self.multihead_att(x,x,x)  #B,Nmax,dmodel


        # x = self.feed_forward(x) 

        
        return x #B,Nmax,dmodel


class Transformer(nn.Module):
    def __init__(self,device,nb_blocks,h,dmodel,d_ff_hidden,dk,dv,dropout = 0.1):
        super(Transformer,self).__init__()
        self.device = device
        blocks = []

        for b in range(nb_blocks):
            blocks.append( EncoderBlock(device,h,dmodel,d_ff_hidden,dk,dv,dropout) )
        self.encoder = nn.Sequential(*blocks)



    def forward(self,x):
        
        x = self.encoder(x)
        return x




