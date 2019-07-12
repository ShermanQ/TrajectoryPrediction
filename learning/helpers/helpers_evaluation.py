from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix,distance
from scipy.stats import norm

from scipy.spatial.distance import euclidean


import matplotlib.image as mpimg
import cv2


import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import torch.nn as nn
import numpy as np
import os 

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.evaluation import Evaluation

# i = i.detach().cpu().numpy()
def revert_scaling_evaluation(offsets_input,scalers_path,i):
    scaler = json.load(open(scalers_path))
    if offsets_input:
        meanx =  scaler["standardization"]["meanx"]
        meany =  scaler["standardization"]["meany"]
        stdx =  scaler["standardization"]["stdx"]
        stdy =  scaler["standardization"]["stdy"]
        

        i[:,:,0] = helpers.revert_standardization(i[:,:,0],meanx,stdx)
        i[:,:,1] = helpers.revert_standardization(i[:,:,1],meany,stdy)
    else:
        min_ =  scaler["normalization"]["min"]
        max_ =  scaler["normalization"]["max"]
        i = helpers.revert_min_max_scale(i,min_,max_)
    return i        

    # i = torch.FloatTensor(i).to(device)        



def types_ohe(types,nb_types):       
    cat = np.arange(1,nb_types+1).reshape(nb_types,1)
    
    ohe = OneHotEncoder(sparse = False,categories = "auto")
    ohe = ohe.fit(cat)

    b,n = types.shape
    # types = types - 1 
    types = ohe.transform(types.reshape(b*n,-1)) 

    types = types.reshape(b,n,nb_types)

    return types

def get_active_mask(mask_target):
    sample_sum = (np.sum(mask_target.reshape(list(mask_target.shape[:2])+[-1]), axis = 2) > 0).astype(int)
                
    active_mask = []
    for b in sample_sum:
        ids = np.argwhere(b.flatten()).flatten()
        active_mask.append(torch.LongTensor(ids))
    return active_mask

def get_data_loader(data_params,data_file,scene,args_net,set_type_test,prepare_param,eval_params):
    dataset = Hdf5Dataset(
            images_path = data_params["prepared_images"],
            hdf5_file= data_file,
            scene_list= [scene], #eval_scenes
            # scene_list= scenes, #eval_scenes

            t_obs=args_net["t_obs"],
            t_pred=args_net["t_pred"],
            set_type = set_type_test, #eval
            use_images = args_net["use_images"],
            data_type = "trajectories",
            # use_neighbors = args_net["use_neighbors"],
            use_neighbors = 1,

            use_masks = 1,
            predict_offsets = args_net["offsets"],
            offsets_input = args_net["offsets_input"],


            predict_smooth= args_net["predict_smooth"],
            smooth_suffix= prepare_param["smooth_suffix"],
            centers = json.load(open(data_params["scene_centers"])),
            padding = prepare_param["padding"],

            augmentation = 0,
            augmentation_angles = [],
            normalize =args_net["normalize"],
            evaluation=1


            )

    data_loader = CustomDataLoader( batch_size = eval_params["batch_size"],shuffle = False,drop_last = False,dataset = dataset,test=0)
    return data_loader

#
#    COnsiders each pair of agent as an interaction
#    Counts the number of problematic interaction
#    averages the percentage over all timesteps
#    returns conflicts coordinates without specifying their timestep
def conflicts(output,threshold = 0.5):
    timesteps = []
    timesteps_disjoint = []

    conflict_points = np.array([])
    for t in range(output.shape[1]):
        points = np.array(output[:,t])
        d = distance_matrix(points,points)

        m = (d < threshold).astype(int) - np.eye(len(points))
        total_count = np.ones_like(m)
        m = np.triu(m,1)
        total_count = np.triu(total_count,1)

        # disjoint (only first trajectory)
        if float(total_count[0].sum()) > 0.:
            conflict_prop_disjoint = m[0].sum() / float(total_count[0].sum())
        else:
            conflict_prop_disjoint = 0
        # joint
        if float(total_count.sum()) > 0.:           
            conflict_prop = m.sum() / float(total_count.sum())
        else:
            conflict_prop = 0

        timesteps.append(conflict_prop)
        timesteps_disjoint.append(conflict_prop_disjoint)


        # select points where conflict happens
        ids = np.unique( np.argwhere(m)[:,0] ) 
        if len(ids) > 0:
            points = points[ids]
            if len(conflict_points) > 0:
                conflict_points = np.concatenate([conflict_points,points], axis = 0)
            else:
                conflict_points = points



    return np.mean(timesteps),np.mean(timesteps_disjoint),timesteps,timesteps_disjoint,conflict_points.tolist()

def dynamic_eval(output,types,dynamics,types_dic,delta_t,dynamic_threshold = 0.0):
    acc_lhood = []
    speed_lhood = []

    for a in range(output.shape[0]):
        coordinates = output[a]
        type_ = types[a]
        type_ = types_dic[str(int(type_+1))]

        speeds = np.array(get_speeds(coordinates,delta_t))
        accelerations = np.array(get_accelerations(speeds,delta_t))
        dynamic_type = dynamics[type_]

        acc_props = norm.pdf(accelerations, loc = dynamic_type["accelerations"]["mean"], scale=dynamic_type["accelerations"]["std"]) 
        speed_props = norm.pdf(accelerations, loc = dynamic_type["speeds"]["mean"], scale=dynamic_type["speeds"]["std"]) 

        acc_lhood.append( acc_props)
        speed_lhood.append(speed_props)

    acc_lhood_disjoint = acc_lhood[0]
    speed_lhood_disjoint = speed_lhood[0]

    return acc_lhood,speed_lhood,acc_lhood_disjoint,speed_lhood_disjoint,np.mean(acc_lhood),np.mean(speed_lhood),np.mean(acc_lhood_disjoint),np.mean(speed_lhood_disjoint)


def get_speed(point1,point2,deltat):
    d = distance.euclidean(point1,point2)
    v = d/deltat
    return v
def get_speeds(coordinates,framerate):
    speeds = []
    for i in range(1,len(coordinates)):
        speed = get_speed(coordinates[i-1],coordinates[i],framerate)
        speeds.append(speed)
    return speeds

def get_acceleration(v1,v2,deltat):
    a = (v2-v1)/deltat
    return a

def get_accelerations(speeds,framerate):
    accelerations = []
    for i in range(1,len(speeds)):
        acceleration = get_acceleration(speeds[i-1],speeds[i],framerate)
        accelerations.append(acceleration)
    return accelerations


# def scene_mask(scene,img_path,class_category,annotations_path):
#         img = mpimg.imread(img_path.format(scene))
#         empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
#         annotations = json.load(open(annotations_path.format(scene)))
#         polygons = []
#         for object_ in annotations["objects"]:
#                 if object_["classTitle"] == class_category:
#                         pts = object_["points"]["exterior"]
#                         a3 = np.array( [pts] ).astype(np.int32)          
#                         cv2.fillPoly( empty_mask, a3, 1 )
#         return empty_mask

# returns array of two masks, first one is pedestrian, second one is wheels
def scene_mask(scene,img_path,annotations_path,spatial_profiles):
        img = mpimg.imread(img_path.format(scene))

        masks = []
        masks_ids = []

        for spatial_profile in spatial_profiles:
        
            empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
            annotations = json.load(open(annotations_path.format(scene)))
            polygons = []
            for object_ in annotations["objects"]:
                    if object_["classTitle"] == spatial_profile:
                            pts = object_["points"]["exterior"]
                            a3 = np.array( [pts] ).astype(np.int32)          
                            cv2.fillPoly( empty_mask, a3, 1 )
            masks.append(empty_mask)
            masks_ids.append(spatial_profiles[spatial_profile])

        arg_ids = np.argsort(masks_ids)
        masks = [masks[i] for i in arg_ids]
        
        return masks

def spatial_conflicts(mask,trajectory_p):
        ctr = 0
        # print(mask.shape)
        for point in trajectory_p:
                #case out of frame
                if point[1] in range(0,mask.shape[0]) and point[0] in range(0,mask.shape[1]):
                    if mask[point[1],point[0]]:
                            ctr += 1
        return ctr / float(len(trajectory_p))

# def spatial_conflicts(mask,trajectory_p):
#         ctr = 0
#         # print(mask.shape)
#         frame_conflicts = np.zeros(len(trajectory_p))
#         for i,point in enumerate(trajectory_p):
#                 #case out of frame
#                 if point[1] in range(0,mask.shape[0]) and point[0] in range(0,mask.shape[1]):
#                     if mask[point[1],point[0]]:
#                         frame_conflicts[i] = 0
                    
#         return frame_conflicts

def spatial_loss(spatial_profile_ids,spatial_masks,outputs,pixel2meters):
    spatial_losses = []
    for id_,trajectory_p in zip(spatial_profile_ids,outputs):
        trajectory_p *= pixel2meters
        trajectory_p = trajectory_p.astype(np.int32)
        res = spatial_conflicts(spatial_masks[id_],trajectory_p)
        spatial_losses.append(res)
    return spatial_losses[0], np.mean(spatial_losses)

# def spatial_loss(spatial_profile_ids,spatial_masks,outputs,pixel2meters):
#     spatial_losses = []
#     frames = []
#     for id_,trajectory_p in zip(spatial_profile_ids,outputs):
#         trajectory_p *= pixel2meters
#         trajectory_p = trajectory_p.astype(np.int32)
#         frame_conflicts = spatial_conflicts(spatial_masks[id_],trajectory_p)
#         frames.append(frame_conflicts)
#     frames = np.array(frames)

#     # joint loss
#     nb_conflicts_per_frame = []
#     for t in range(frames.shape[1]):
#         frame = frames[:,t]
#         nb_conflicts_per_frame.append(np.sum(frame))
#     # disjoint loss
#     nb_conflicts_per_frame_disjoint = []
#     for t in range(frames.shape[1]):
#         frame = frames[0,t]
#         nb_conflicts_per_frame_disjoint.append(np.sum(frame))

#     return np.mean(nb_conflicts_per_frame_disjoint), np.mean(nb_conflicts_per_frame) ,nb_conflicts_per_frame_disjoint, nb_conflicts_per_frame

# for t in range(output.shape[1]):
#     points = np.array(output[:,t])


def get_factor(scene,correspondences_trajnet,correspondences_manual):
    if scene in correspondences_trajnet:
        row = correspondences_trajnet[scene]
        pixel2meter_ratio = row["pixel2meter"]
        meter2pixel_ratio = 1/pixel2meter_ratio
    else:
        row = correspondences_manual[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        pixel2meter_ratio = meter_dist/float(pixel_dist)
        meter2pixel_ratio = float(pixel_dist)/meter_dist

    return meter2pixel_ratio


def predict_neighbors_disjoint(inputs,types,active_mask,points_mask,imgs,net,device):
    b,n,s,i = points_mask[0].shape
    b,n,p,i = points_mask[1].shape

    # permute every samples
    batch_perms = []
    batch_p0 = []
    batch_p1 = []
    for batch_element,p0,p1 in zip(inputs,points_mask[0],points_mask[1]):
        batch_element_perms = []  
        batch_p0_perms = []
        batch_p1_perms = []

        ids_perm = np.arange(n)

        for ix in range(n):
            ids_perm = np.roll(ids_perm,-ix)
            batch_element_perms.append(batch_element[torch.LongTensor(ids_perm)])
            batch_p0_perms.append(p0[ids_perm])
            batch_p1_perms.append(p1[ids_perm])
        
        
        batch_element_perms = torch.stack(batch_element_perms)
        batch_perms.append(batch_element_perms)
        batch_p0_perms = np.array(batch_p0_perms)
        batch_p0.append(batch_p0_perms)
        batch_p1_perms = np.array(batch_p1_perms)
        batch_p1.append(batch_p1_perms)

    # b,n,s,i -> b,n,n,s,i
    batch_perms = torch.stack(batch_perms)
    batch_p0 = np.array(batch_p0)
    batch_p1 = np.array(batch_p1)

    # b,n,n,s,i -> b*n,n,s,i
    batch_perms = batch_perms.view(-1,n,s,i)
    batch_p0 = batch_p0.reshape(-1,n,s,i)
    batch_p1 = batch_p1.reshape(-1,n,p,i)

    # save inputs
    inputs_temp = inputs
    points_mask_temp = points_mask
    active_mask_temp = active_mask

    # new inputs from permutations
    inputs = batch_perms
    points_mask = (batch_p0,batch_p1)
    active_mask = torch.arange(inputs.size()[0]*inputs.size()[1]).to(device)

    # prediction
    outputs = net((inputs,types,active_mask,points_mask,imgs))

    # reset inputs
    inputs = inputs_temp
    points_mask = points_mask_temp
    active_mask = active_mask_temp

    # select outputs
    outputs = outputs[:,0]
    outputs = outputs.view(b,n,p,i)
    return outputs,inputs,types,active_mask,points_mask
        
def predict_naive(inputs,types,active_mask,points_mask,imgs,net,device):
    b,n,s,i = points_mask[0].shape
    b,n,p,i = points_mask[1].shape

    inputs = inputs.view(-1,s,i).unsqueeze(1)
    types = types.view(-1).unsqueeze(1)
    points_mask[0] = np.expand_dims(points_mask[0].reshape(-1,s,i),1)
    points_mask[1] = np.expand_dims(points_mask[1].reshape(-1,p,i),1)
    imgs = imgs.repeat(n,1,1,1)
    # prediction
    outputs = net((inputs,types,active_mask,points_mask,imgs))

    # b*n,s,i -> b,n,s,i
    outputs = outputs.squeeze(1).view(b,n,p,i)
    inputs = inputs.squeeze(1).view(b,n,s,i)
    types = types.squeeze(1).view(b,n)
    points_mask[0] = points_mask[0].squeeze(1).reshape(b,n,s,i)
    points_mask[1] = points_mask[1].squeeze(1).reshape(b,n,p,i)

    return outputs,inputs,types,active_mask,points_mask
    