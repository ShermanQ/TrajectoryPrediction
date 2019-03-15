import cv2
import numpy as np 


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
            for id_ in frame["ids"]:
                # if id_ != "frame":
                coordinates = frame["ids"][id_]["coordinates"]

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

# def get_scene_image_size(min_,max_,factor_div = 2.0, E = 0):
def get_scene_image_size(min_,max_):
    s = np.subtract(max_,min_)
    return s[0],s[1]

def get_number_object(file_path):
    dict_ = {}
    with open(file_path) as scene:
        for line in scene:
            line = line.split(",")
            id_ = line[3]

            dict_[id_] = 1
    return len(dict_.keys())


def plot_current_frame(frame,img1,color_dict,new_color_index,factor_div,offset,display_box = True):

   
    for id_ in frame["ids"]:
        # if id_ != "frame":

        if id_ not in color_dict:
            # color_dict[id_] = tuple(colors[new_color_index][:3] * 255)
            color_dict[id_] = [int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )]

            
            new_color_index += 1

        coordinates = frame["ids"][id_]["coordinates"]
        bbox = frame["ids"][id_]["bbox"]
        bbox = [[bbox[0],bbox[1]],[bbox[2],bbox[3]]]
        # print(bbox)
        # print(type(color_dict[id_][0]))
        # print(coordinates)
        # print(tuple([int(p/factor_div) for p in coordinates]))
        # print("----")
        scaled_coordinates = [p/factor_div for p in coordinates]
        offset_coordinates = np.add(scaled_coordinates,offset).tolist()
        int_coordinates = [int(p) for p in offset_coordinates]
        cv2.circle(img1,tuple(int_coordinates), 5, tuple(color_dict[id_]), -1)
        cv2.putText(img1, str(id_), tuple(np.add(int_coordinates,[10,10]).tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(color_dict[id_]), 2, cv2.LINE_AA)

        if display_box and bbox[0] != -1:
            scaled_bbox = [[c/factor_div for c in p]for p in bbox]
            offset_bbox = [ np.add(p,offset).tolist() for p in scaled_bbox]
            int_bbox = [[int(c) for c in p ]for p in offset_bbox]
            
            cv2.rectangle(img1,tuple(int_bbox[0]),tuple(int_bbox[1]),tuple(color_dict[id_]),3)
    return img1,color_dict,new_color_index

def plot_frame_number(h,w, img1,frame, size = 0.8, offset = [100,25], color = (255, 255, 255),  font = cv2.FONT_HERSHEY_SIMPLEX):
    
                
    text_pos = tuple(np.subtract([h,w],offset))
    
    cv2.putText(img1, str(frame["frame"]), text_pos, font, size, color, 2, cv2.LINE_AA)

    return img1

def plot_scene_name(h,w, img1,frame, size = 0.8, offset = [400,25], color = (255, 255, 255),  font = cv2.FONT_HERSHEY_SIMPLEX):
    
    scene = frame["ids"][[key for key in frame["ids"].keys()][0]]["scene"]            
    text_pos = tuple(np.subtract([h,w],offset))
    
    cv2.putText(img1, scene, text_pos, font, size, color, 2, cv2.LINE_AA)

    return img1

def plot_last_steps(img1,frame,last_frames,color_dict,frequency_points = 5,factor_div = 2.0, line_thickness = 1,point_thickness = 2,offset = [0.,0.]):
    for id_ in frame["ids"]:
        # if id_ != "frame":
        points = []
        for f in last_frames:
            if id_ in f["ids"]:

                scaled_coordinates = [p/factor_div for p in f["ids"][id_]["coordinates"]]
                offset_coordinates = np.add(scaled_coordinates,offset).tolist()
                int_coordinates = [int(p) for p in offset_coordinates]

                points.append(int_coordinates)
                

        for i,p in enumerate(points):
            if i % frequency_points == 0:
                cv2.circle(img1,tuple(p), point_thickness, tuple(color_dict[id_]), -1)

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img1,[pts],False,tuple(color_dict[id_]),thickness = line_thickness)
    return img1