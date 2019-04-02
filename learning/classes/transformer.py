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
        return att

