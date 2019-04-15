import sys
import json
import csv
import numpy as np
import pandas as pd
import helpers
from scipy.spatial.distance import euclidean
from skimage import io,transform,util
import os



class SceneCenters():
    def __init__(self,preprocessing_params,data,prepare_params):
        preprocessing_params = json.load(open(preprocessing_params)) 
        prepare_params = json.load(open(prepare_params)) 
        data = json.load(open(data))

        self.correspondences = json.load(open(data["sdd_pixel2meters"]))


        self.original_image = data["original_images"] + "{}.jpg"
        self.destination_json = data["scene_centers"]
  
    def get_centers(self, scenes):
        centers = {}
        for scene in scenes:
            center = self.scene_center(scene)
            centers[scene] = center
        json.dump(centers,open(self.destination_json,"w"))
        return centers
    def scene_center(self,scene):
        img = io.imread(self.original_image.format(scene))
        self.__get_factor(scene)
        h,w,_ = img.shape
        
        h = int(self.pixel2meter_ratio * float(h)/2.0*100) /100.0 
        w = int(self.pixel2meter_ratio * float(w)/2.0*100) /100.0 
        
        center = (h,w)

        return center

    def __get_factor(self,scene):
        row = self.correspondences[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        self.pixel2meter_ratio = meter_dist/float(pixel_dist)