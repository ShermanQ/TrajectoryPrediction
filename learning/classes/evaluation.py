# Dataloader classique si use_neighbors renvoie B,N,T,2 sinon renvoie B,1,T,2
# Dans evaluation si use_neighbors expand(0) B,1,N,T,2 sinon expand()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import json 
import torch
import sys
import os
from scipy.spatial import distance_matrix
from scipy.stats import norm
from scipy.spatial.distance import euclidean



# from ../helpers import get_speeds,get_accelerations

import time

def get_speed(point1,point2,deltat):
    d = euclidean(point1,point2)
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


class Evaluation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"
        self.reports_dir = self.data_params["reports_evaluation"] + "{}/"

        self.dynamics = json.load(open(self.data_params["dynamics"]))
        self.delta_t = 1.0/float(self.prepare_params["framerate"])

        self.types_dic = self.prepare_params["types_dic_rev"]

        self.dynamic_threshold = self.eval_params["dynamic_threshold"]
