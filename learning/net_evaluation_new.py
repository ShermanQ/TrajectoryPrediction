
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix,distance
from scipy.stats import norm

from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance


import matplotlib.image as mpimg
import cv2

import copy

import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import helpers.helpers_evaluation as helpers_evaluation
import torch.nn as nn
import numpy as np
import os 
import copy
import matplotlib.pyplot as plt

def speeds_distance(scene_files,types_dic,delta_t):
    speed_real_distribution = {}
    speed_predicted_distribution = {}

    speed_real_distribution["global"] = []
    speed_predicted_distribution["global"] = []


    for scene_file in scene_files:
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            type_str = types_dic[str(int(types[0]))]
            if type_str not in speed_predicted_distribution:
                speed_predicted_distribution[type_str] = []
            if type_str not in speed_real_distribution:
                speed_real_distribution[type_str] = []

            speeds_labels = helpers_evaluation.get_speeds(labels[0],delta_t)
            speeds_outputs = helpers_evaluation.get_speeds(outputs[0],delta_t)

            speed_real_distribution[type_str] += speeds_labels
            speed_predicted_distribution[type_str] += speeds_outputs
            speed_predicted_distribution["global"] += speeds_outputs
            speed_real_distribution["global"] += speeds_labels

    results = {}
    for type_ in speed_real_distribution:
        results[type_] = wasserstein_distance(speed_predicted_distribution[type_],speed_real_distribution[type_])
    return results

def accelerations_distance(scene_files,types_dic,delta_t):
    acceleration_real_distribution = {}
    acceleration_predicted_distribution = {}

    acceleration_real_distribution["global"] = []
    acceleration_predicted_distribution["global"] = []


    for scene_file in scene_files:
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            type_str = types_dic[str(int(types[0]))]
            if type_str not in acceleration_predicted_distribution:
                acceleration_predicted_distribution[type_str] = []
            if type_str not in acceleration_real_distribution:
                acceleration_real_distribution[type_str] = []

            speeds_labels = helpers_evaluation.get_speeds(labels[0],delta_t)
            speeds_outputs = helpers_evaluation.get_speeds(outputs[0],delta_t)

            accelerations_labels = helpers_evaluation.get_accelerations(speeds_labels,delta_t)
            accelerations_outputs = helpers_evaluation.get_accelerations(speeds_outputs,delta_t)

            acceleration_real_distribution[type_str] += accelerations_labels
            acceleration_predicted_distribution[type_str] += accelerations_outputs
            acceleration_predicted_distribution["global"] += accelerations_outputs
            acceleration_real_distribution["global"] += accelerations_labels

    results = {}
    for type_ in acceleration_real_distribution:
        results[type_] = wasserstein_distance(acceleration_predicted_distribution[type_],acceleration_real_distribution[type_])
    return results

# python model_evaluation.py parameters/data.json parameters/prepare_training.json parameters/model_evaluation.json 
def main():
    args = sys.argv


    eval_params = json.load(open("parameters/model_evaluation.json"))
    data_params = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))

    prepare_param = json.load(open("parameters/prepare_training.json"))


    # spatial loss variables
    spatial_annotations = data_params["spatial_annotations"]+"{}.jpg.json"
    images = data_params["original_images"]+"{}.jpg"
    types_to_spatial = eval_params["types2spatial"]
    spatial_profiles = eval_params["spatial_profiles"]
    correspondences_trajnet = json.load(open(data_params["pixel2meters_trajnet"]))
    correspondences_manual = json.load(open(data_params["pixel2meters_manual"]))

    #########################
    
    
#   social original
# ade
# fde
#spatial check que tout va bien/ seulement sur la première traj

    data_file = data_params["hdf5_file"]
    report_name = eval_params["report_name"]

    dir_name = data_params["reports_evaluation"] + "{}/".format(report_name)
    sub_dir_name = data_params["reports_evaluation"] + "{}/scene_reports/".format(report_name) 

    # scenes = [ f.split("_")[0]  for f in  os.listdir(sub_dir_name) if "json" in f]
    # print(scenes)
    # for scene in scenes:
    #     spatial_masks = helpers_evaluation.scene_mask(scene,images,spatial_annotations,spatial_profiles)
    #     print(spatial_masks[0])
    #     plt.imshow(spatial_masks[0])
    #     plt.show()
        # cv2.imshow("imgs",spatial_masks[0])
        # cv2.waitKey()

    types_dic = prepare_param["types_dic_rev"]
    if os.path.exists(dir_name) and os.path.exists(sub_dir_name):
        scene_files = [ sub_dir_name+f  for f in  os.listdir(sub_dir_name) if "json" in f]
        # speed_results = speeds_distance(scene_files,types_dic,1.0/float(prepare_param["framerate"]))
        # acceleration_results = accelerations_distance(scene_files,types_dic,1.0/float(prepare_param["framerate"]))
        # print(acceleration_results)
        # print(speed_results)

        # social disjoint, traject7ire prédite et agents voisins groundtruth
        # conflicts_distrib_results = get_distrib_conflicts(scene_files)
        
        social_results = social_conflicts(scene_files)
        print(social_results)


def social_conflicts(scene_files):
    social_results = {}
    conflict_thresholds = [0.1,0.5,1.0]
    social_results["global"] = {}
    for thresh in conflict_thresholds:
        social_results["global"]["joint_"+str(thresh)] = []
        social_results["global"]["disjoint_"+str(thresh)] = []
        social_results["global"]["groundtruth_"+str(thresh)] = []


    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]
        social_results[scene] = {}
        for thresh in conflict_thresholds:
            social_results[scene]["joint_"+str(thresh)] = []
            social_results[scene]["disjoint_"+str(thresh)] = []
            social_results[scene]["groundtruth_"+str(thresh)] = []
        print(scene)
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]

    # social loss
            social_sample = copy.copy(labels)
            social_sample[0] = outputs[0]
            social_sample = np.array(social_sample)
            labels = np.array(labels)
            outputs = np.array(outputs)    

            # social loss
            for thresh in conflict_thresholds:
                # print(thresh)
                frames_joint = conflicts(outputs,thresh)
                frames_disjoint = conflicts(social_sample,thresh)
                frames_gt = conflicts(labels,thresh)


                social_results["global"]["joint_"+str(thresh)] += frames_joint
                social_results["global"]["disjoint_"+str(thresh)] += frames_disjoint
                social_results["global"]["groundtruth_"+str(thresh)] += frames_gt

                social_results[scene]["joint_"+str(thresh)] += frames_joint
                social_results[scene]["disjoint_"+str(thresh)] += frames_disjoint
                social_results[scene]["groundtruth_"+str(thresh)] += frames_gt
        for thresh in conflict_thresholds:
            social_results[scene]["joint_"+str(thresh)] = np.mean(social_results[scene]["joint_"+str(thresh)])
            social_results[scene]["disjoint_"+str(thresh)] = np.mean(social_results[scene]["disjoint_"+str(thresh)])
            social_results[scene]["groundtruth_"+str(thresh)] = np.mean(social_results[scene]["groundtruth_"+str(thresh)])

    for thresh in conflict_thresholds:
        social_results["global"]["joint_"+str(thresh)] = np.mean(social_results["global"]["joint_"+str(thresh)])
        social_results["global"]["disjoint_"+str(thresh)] = np.mean(social_results["global"]["disjoint_"+str(thresh)])
        social_results["global"]["groundtruth_"+str(thresh)] = np.mean(social_results["global"]["groundtruth_"+str(thresh)])
    return social_results

                    



        # print(wasserstein_distance(distrib_pred,distrib_real))
def conflicts(trajectories,threshold = 0.5):
    timesteps = []
    for t in range(trajectories.shape[1]):
        points = np.array(trajectories[:,t])
        conflict_prop = conflicts_frame(points,threshold)
        timesteps.append(conflict_prop)
    return timesteps

def conflicts_frame(points,threshold):
    d = distance_matrix(points,points)

    m = (d < threshold).astype(int) - np.eye(len(points))
    total_count = np.ones_like(m)
    m = np.triu(m,1)
    total_count = np.triu(total_count,1)


    if float(total_count.sum()) > 0.:           
        conflict_prop = m.sum() / float(total_count.sum())
    else:
        conflict_prop = 0
    return conflict_prop
        


def get_distrib_conflicts(scene_files):
    distrib_pred_disjoint = {"global":[]}
    distrib_pred = {"global":[]}
    distrib_real = {"global":[]}
    results = {}


    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]
        # print(scene)

        distrib_pred[scene] = []
        distrib_pred_disjoint[scene] = []
        distrib_real[scene] = []



        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            social_sample = copy.copy(labels)
            social_sample[0] = outputs[0]
            social_sample = np.array(social_sample)
            labels = np.array(labels)
            outputs = np.array(outputs)


            distances_pred_disjoint = get_distances_agents_interval(social_sample)
            distances_pred = get_distances_agents_interval(outputs)
            distances_real = get_distances_agents_interval(labels)

            distrib_pred_disjoint["global"] += distances_pred_disjoint
            distrib_pred["global"] += distances_pred
            distrib_real["global"] += distances_real

            distrib_pred_disjoint[scene] += distances_pred_disjoint
            distrib_pred[scene] += distances_pred
            distrib_real[scene] += distances_real
        results[scene] = {}
        results[scene]["disjoint"] = wasserstein_distance(distrib_pred_disjoint[scene],distrib_real[scene])
        results[scene]["joint"] = wasserstein_distance(distrib_pred[scene],distrib_real[scene])



    results["global"] = {}
    results["global"]["disjoint"] = wasserstein_distance(distrib_pred_disjoint["global"],distrib_real["global"])
    results["global"]["joint"] = wasserstein_distance(distrib_pred["global"],distrib_real["global"])

    return results
    

def get_distances_agents_frame(points):
    d = distance_matrix(points,points)
    d = np.triu(d,1).flatten()
    distances = [e for e in d if e != 0.]
    return distances 
def get_distances_agents_interval(trajectories):
    distances_interval = []
    for t in range(trajectories.shape[1]):
        points = np.array(trajectories[:,t])
        distances = get_distances_agents_frame(points)
        distances_interval += distances
    return distances_interval



if __name__ == "__main__":
    main()