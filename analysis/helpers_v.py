import cv2
import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extractors.helpers as helpers

import csv
import json
import matplotlib.cm as cm
from collections import deque


def find_bounding_coordinates(frames_path):
    minx = 10e30
    miny = 10e30
    maxx = 10e-30
    maxy = 10e-30

    with open(frames_path) as frames:
        for frame in frames:
            frame = json.loads(frame)
            for id_ in frame:
                if id_ != "frame":
                    coordinates = frame[id_]["coordinates"]

                    x = coordinates[0]
                    y = coordinates[1]

                    if x > maxx:
                        maxx = x
                    if y > maxy:
                        maxy = y
                    if x < minx:
                        minx = x
                    if y < miny:
                        miny = y
    return [minx,miny],[maxx,maxy]

def get_scene_image_size(min_,max_,factor_div = 2.0, E = 100):
    fx = max_[0] + E
    fy = max_[1] + E
    if min_[0] < 0:
        fx -= min_[0]
    if min_[1] < 0:
        fy -= min_[1]
    return int(fx/factor_div),int(fy/factor_div)
def get_number_object(file_path):
    dict_ = {}
    with open(file_path) as scene:
        for line in scene:
            line = line.split(",")
            id_ = line[3]

            dict_[id_] = 1
    return len(dict_.keys())


def plot_current_frame(frame,img1,color_dict,new_color_index,factor_div):

   
    for id_ in frame:
        if id_ != "frame":

            if id_ not in color_dict:
                # color_dict[id_] = tuple(colors[new_color_index][:3] * 255)
                color_dict[id_] = [int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )]

                
                new_color_index += 1

            coordinates = frame[id_]["coordinates"]
            # print(type(color_dict[id_][0]))
            
            
            cv2.circle(img1,tuple([int(p/factor_div) for p in coordinates]), 5, tuple(color_dict[id_]), -1)
    return img1,color_dict,new_color_index

def plot_frame_number(h,w, img1,frame, size = 0.8, offset = [100,25], color = (255, 255, 255),  font = cv2.FONT_HERSHEY_SIMPLEX):
    
                
    text_pos = tuple(np.subtract([h,w],offset))
    
    cv2.putText(img1, str(frame["frame"]), text_pos, font, size, color, 2, cv2.LINE_AA)

    return img1

def plot_last_steps(img1,frame,last_frames,color_dict,frequency_points = 5,factor_div = 2.0, line_thickness = 1,point_thickness = 2):
    for id_ in frame:
        if id_ != "frame":
            points = []
            for f in last_frames:
                if id_ in f:
                    points.append([int(p/factor_div) for p in f[id_]["coordinates"]])
                    

            for i,p in enumerate(points):
                if i % frequency_points == 0:
                    cv2.circle(img1,tuple(p), point_thickness, tuple(color_dict[id_]), -1)

            pts = np.array(points, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img1,[pts],False,tuple(color_dict[id_]),thickness = line_thickness)
    return img1