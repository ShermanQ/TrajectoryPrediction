import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load


def split_train_eval_test(ids,train_scenes,test_scenes, eval_prop = 0.8):
    test_ids,train_ids,eval_ids = [],[],[]
    train = {}

    for id_ in ids:
        scene = id_.split("_")[0]
        if scene in test_scenes:
            test_ids.append(id_)
        elif scene in train_scenes:
            if scene not in train:
                train[scene] = []
            train[scene].append(id_)
    
    for key in train:
        nb_scene_samples = len(train[key])
        nb_train = int(eval_prop*nb_scene_samples)

        train_ids += train[key][:nb_train]
        eval_ids += train[key][nb_train:]

    return train_ids,eval_ids,test_ids


def revert_scaling(ids,labels,outputs,scalers_root,multiple_scalers = 1):
    if multiple_scalers:
        scaler_ids = ["_".join(id_.split("_")[:-1]) for id_ in ids]
        scalers_path = [scalers_root + id_ +".joblib" for id_ in scaler_ids]
       
        scaler_sample = {}
        for scaler in scalers_path:
            if scaler not in scaler_sample:
                scaler_sample[scaler] = []

                for i,scaler1 in enumerate(scalers_path):
                    if scaler == scaler1:
                        scaler_sample[scaler].append(i)

        for scaler_id in scaler_sample:
            scaler = load(scaler_id)
            samples_ids = scaler_sample[scaler_id]

            sub_labels_torch = labels[samples_ids]
            # b,a,s,i = sub_labels.size()

            sub_labels = sub_labels_torch.contiguous().view(-1,1).cpu().numpy()
            inv_sub_labels = torch.FloatTensor(scaler.inverse_transform(sub_labels)).view(sub_labels_torch.size()).cuda()
            labels[samples_ids] = inv_sub_labels

            sub_outputs_torch = outputs[samples_ids]
            # b,a,s,i = sub_outputs.size()

            sub_outputs = sub_outputs_torch.contiguous().view(-1,1).cpu().detach().numpy()
            inv_sub_outputs = torch.FloatTensor(scaler.inverse_transform(sub_outputs)).view(sub_outputs_torch.size()).cuda()
            outputs[samples_ids] = inv_sub_outputs
        return labels,outputs
    else:

        scaler = load(scalers_root)
        torch_labels = labels.contiguous().view(-1,1).cpu().numpy()
        torch_outputs = outputs.contiguous().view(-1,1).cpu().detach().numpy()

        non_zeros_labels = np.argwhere(torch_labels.reshape(-1))
        non_zeros_outputs = np.argwhere(torch_outputs.reshape(-1))

         
        torch_labels[non_zeros_labels] = np.expand_dims( scaler.inverse_transform(torch_labels[non_zeros_labels].squeeze(-1)) ,axis = 1)
        
        torch_outputs[non_zeros_outputs] = np.expand_dims( scaler.inverse_transform(torch_outputs[non_zeros_outputs].squeeze(-1)),axis = 1)

        inv_labels = torch.FloatTensor(torch_labels).cuda()
        inv_outputs = torch.FloatTensor(torch_outputs).cuda()

        inv_labels = inv_labels.view(labels.size())
        inv_outputs = inv_outputs.view(outputs.size())


        return inv_labels,inv_outputs


def mask_loss(targets):
    b,a = targets.shape[0],targets.shape[1]
    mask = targets.reshape(b,a,-1)
    mask = np.sum(mask,axis = 2)
    mask = mask.reshape(-1)
    mask = np.argwhere(mask).reshape(-1)
    return mask    


def ade_loss(outputs,targets):

    
    outputs = outputs.contiguous().view(-1,2)
    targets = targets.contiguous().view(-1,2)
    mse = nn.MSELoss(reduction= "none")

    mse_loss = mse(outputs,targets )
    mse_loss = torch.sum(mse_loss,dim = 1 )
    mse_loss = torch.sqrt(mse_loss )
    mse_loss = torch.mean(mse_loss )

    return mse_loss

def fde_loss(outputs,targets):

    

    outputs = outputs[:,-1,:]
    targets = targets[:,-1,:]
    mse = nn.MSELoss(reduction= "none")

    mse_loss = mse(outputs,targets )
    mse_loss = torch.sum(mse_loss,dim = 1 )
    mse_loss = torch.sqrt(mse_loss )
    mse_loss = torch.mean(mse_loss )

    return mse_loss