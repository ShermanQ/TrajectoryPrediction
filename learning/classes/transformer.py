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
    def __init__(self,dropout = 0.1):
        super(ScaledDotProduct, self).__init__()

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

    def get_mask(self,q,k):
        # compute mask put one where the dot product conv_features*conv_features.T is 0.
        # and the sum over a given row of the resulting matrice is gt 1
        Nmax = q.size()[1]
        dot = torch.bmm(q, torch.transpose(k,2,1))
        mask = (dot == 0) & (torch.sum(dot,dim = 1) > 0.).unsqueeze(2).repeat(1,1,Nmax)
        return mask

        

class AttentionHead(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,dmodel,dk,dv,dropout = 0.1):
        super(AttentionHead,self).__init__()
        self.q_projection = nn.Linear(dmodel,dk)
        self.k_projection = nn.Linear(dmodel,dk)
        self.v_projection = nn.Linear(dmodel,dv)
        self.dot_attention = ScaledDotProduct(dropout)

    def forward(self,q,k,v):
        Q = self.q_projection(q)
        K = self.k_projection(k)
        V = self.v_projection(v)
        att = self.dot_attention(Q,K,V,self.dot_attention.get_mask(Q,K))
        return att #B,Nmax,dv

class MultiHeadAttention(nn.Module):
    def __init__(self,h,dmodel,dk,dv,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        assert dmodel == h*dv
        assert dk == dv

        self.heads  = []
        for i in range(h):
            self.heads.append(AttentionHead(dmodel,dk,dv,dropout))  # inefficient#############################
        

        self.multihead_projection = nn.Linear(h*dv,dmodel)

    def forward(self,q,k,v):
        atts = [] #H,Nmax,dv
        for head in self.heads:
            atts.append(head(q,k,v))#B,Nmax,dv

        atts = torch.cat(atts,dim = 2) #B,Nmax,dv * h
        out = self.multihead_projection(atts) #B,Nmax,dmodel

        return out

        
        

class EncoderBlock(nn.Module):
    def __init__(self,h,dmodel,d_ff_hidden,dk,dv,dropout = 0.1):
        super(EncoderBlock,self).__init__()
        self.multihead_att = MultiHeadAttention(h,dmodel,dk,dv,dropout )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dmodel,d_ff_hidden),
            nn.ReLU(),
            nn.Linear(d_ff_hidden,dmodel)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(dmodel)
        self.norm_layer2 = nn.LayerNorm(dmodel)


    def forward(self,x):
        x = self.norm_layer1( x + self.dropout(self.multihead_att(x,x,x)) ) #B,Nmax,dmodel

        x = self.norm_layer2( x + self.dropout( self.feed_forward(x) ) )

        return x #B,Nmax,dmodel


class Encoder(nn.Module):
    def __init__(self,nb_blocks,h,dmodel,d_ff_hidden,dk,dv,dropout = 0.1):
        super(Encoder,self).__init__()

        blocks = []

        for b in range(nb_blocks):
            blocks.append( EncoderBlock(h,dmodel,d_ff_hidden,dk,dv,dropout) )
        self.encoder = nn.Sequential(*blocks)



    def forward(self,x):
        x = self.encoder(x)
        return x




