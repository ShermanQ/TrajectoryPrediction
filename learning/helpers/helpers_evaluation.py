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



    return np.mean(timesteps),np.mean(timesteps_disjoint),conflict_points.tolist()

def dynamic_eval(output,types,dynamics,types_dic,delta_t,dynamic_threshold = 0.0):
    count_per_traj_speed = []
    count_per_traj_acc = []

    for a in range(output.shape[0]):
        coordinates = output[a]
        type_ = types[a]
        type_ = types_dic[str(int(type_+1))]

        speeds = get_speeds(coordinates,delta_t)
        accelerations = get_accelerations(speeds,delta_t)
        dynamic_type = dynamics[type_]

        acc_props = norm.pdf(accelerations, loc = dynamic_type["accelerations"]["mean"], scale=dynamic_type["accelerations"]["std"]) 
        speed_props = norm.pdf(accelerations, loc = dynamic_type["speeds"]["mean"], scale=dynamic_type["speeds"]["std"]) 


        nb_outliers_speeds = (speed_props < dynamic_threshold).astype(int).sum()    
        nb_outliers_accs = (acc_props < dynamic_threshold).astype(int).sum()        

        percentage_outlier_points_speed = nb_outliers_speeds/len(speeds) * 100
        percentage_outlier_points_accs = nb_outliers_accs/len(accelerations) * 100

        count_per_traj_speed.append(percentage_outlier_points_speed)
        count_per_traj_acc.append(percentage_outlier_points_accs)

    
    acc_len = np.mean(count_per_traj_acc)
    speed_len = np.mean(count_per_traj_speed)

    return speed_len,acc_len,count_per_traj_speed[0],count_per_traj_acc[0]


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

def spatial_loss(spatial_profile_ids,spatial_masks,outputs,pixel2meters):
    spatial_losses = []
    for id_,trajectory_p in zip(spatial_profile_ids,outputs):
        trajectory_p *= pixel2meters
        trajectory_p = trajectory_p.astype(np.int32)
        res = spatial_conflicts(spatial_masks[id_],trajectory_p)
        spatial_losses.append(res)
    return spatial_losses[0], np.mean(spatial_losses)

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

