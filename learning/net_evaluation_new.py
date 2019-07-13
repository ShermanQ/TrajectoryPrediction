
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

#   1.0/float(prepare_param["framerate"]

    data_file = data_params["hdf5_file"]
    report_name = eval_params["report_name"]

    dir_name = data_params["reports_evaluation"] + "{}/".format(report_name)
    sub_dir_name = data_params["reports_evaluation"] + "{}/scene_reports/".format(report_name) 

    types_dic = prepare_param["types_dic_rev"]
    if os.path.exists(dir_name) and os.path.exists(sub_dir_name):
        scene_files = [ sub_dir_name+f  for f in  os.listdir(sub_dir_name) if "json" in f]
        speed_results = speeds_distance(scene_files,types_dic,1.0/float(prepare_param["framerate"]))
        acceleration_results = accelerations_distance(scene_files,types_dic,1.0/float(prepare_param["framerate"]))
        print(acceleration_results)
        print(speed_results)


if __name__ == "__main__":
    main()