import os
import numpy as np

import sys
import csv
import time


"""
    Get the directories contained in a directory
    path: directory path
    lower: set the names to lower case
    ordered: order directory names lexygraphically
    descending: descending order for directory names
"""
def get_dir_names(path,lower = True,ordered = True,descending = False):
    dir_names = []
    dirs = os.listdir(path)
    if ordered:
        dirs = sorted(dirs,key = str, reverse = descending)
    for x in dirs:
        if lower:
            x = x.lower()
        dir_names.append(x)
    return dir_names


"""
    Parse and add scene specific informations to a line
    line : a line of the original SDD dataset
    scene: scene name + subscene number
    dataset: name of the dataset
"""
def parse_line(line,scene, dataset ):

    dict_type = {
        "\"Biker\"\n" : "bicycle", 
        "\"Pedestrian\"\n" : "pedestrian", 
        "\"Cart\"\n" : "cart", 
        "\"Car\"\n": "car", 
        "\"Bus\"\n" : "bus", 
        "\"Skater\"\n": "skate" }
    line = line.split(" ")

    new_line = []    
    
    xa = float(line[1])
    ya = float(line[2])
    xb = float(line[3])
    yb = float(line[4])

    x = str((xa + xb)/2)
    y = str((ya+yb)/2 )

    new_line.append(dataset) # dataset label
    new_line.append(scene)   # subscene label
    new_line.append(line[5]) #frame
    new_line.append(line[0]) #id

    new_line.append(x) #x
    new_line.append(y) #y
    new_line.append(line[1]) # xmin. The top left x-coordinate of the bounding box.
    new_line.append(line[2]) # ymin The top left y-coordinate of the bounding box.
    new_line.append(line[3]) # xmax. The bottom right x-coordinate of the bounding box.
    new_line.append(line[4]) # ymax. The bottom right y-coordinate of the bounding box.

    new_line.append(dict_type[line[9]]) # label type of agent    

    return new_line

"""
    Intersection over Union between two bounding boxes
    box = [xtopleft,ytopleft,xbottomright,ybottomright]
""" 

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


"""
    Remove a list of chars from string
"""
def remove_char(chars,string):
    # chars_to_remove = ['{','}',' ','\'']
    sc = set(chars)
    string = ''.join([c for c in string if c not in sc])
    return string

"""
    In bad extractor parse row from text file obtained
    after parsing the init cpkl file
"""
def parse_init_line(row):
    trajectory_id = int(row[0])
    class_id = row[1]
    box = remove_char(['[',']'],row[2]).split(",")
    box = [float(b) for b in box]
    frame = int(row[3])
    return trajectory_id,class_id,box,frame

"""
    In bad extractor parse row from text file obtained
    after parsing the boxes cpkl file
"""

def parse_boxes_line(row):

    nb_points = 4 
    
    box = remove_char(['[',']'],row[0]).split(",")
    box = [float(b) for b in box]
    boxes = []
    for j in range(int(len(box)/nb_points)):
        sub_box = []
        for i in range(nb_points):
            sub_box.append(float(box[nb_points * j + i]))
        boxes.append(sub_box)

    frame = int(row[1])
    return boxes,frame



"""
    Get the first timestep for a scene in fsc dataset
"""
def frame_corrector(subscene_path):
    with open(subscene_path,"r") as a:

        start_frame = time.time()
        
        for k,line in enumerate(a):

            frame = float(line.split(" ")[0])
            # if k == 0:
                # start_frame = frame
            if frame < start_frame:
                start_frame = frame
  
    return start_frame


"""
    Parse and add scene specific informations to a line
    line : a line of the original fsc dataset
    scene: scene name + subscene number
    dataset: name of the dataset
    min_step: framrate * 100
    start_frame: first timestamp of the scene
"""   

def parse_line_fsc(line,scene_name, dataset_name,min_step,start_frame):

    undefined = -1

    line = line.split(" ")

    timestamp = round(float(line[0]),2)*100
    start_frame = round(start_frame,2)*100
    dt = timestamp - start_frame

    frame = int(dt/min_step)
    id_ = line[1]
    
    r = -1
    theta = -1
    x = -1
    y = -1

    if "" != line[3] and "" != line[-1]: #domaine de definition dans le fichier et domaine d'application de la formule?
        
        r = float(line[3])     
        theta = float(line[-1])

        x = r * np.cos(theta)
        y = r * np.sin(theta)

    new_line = []

    new_line.append(dataset_name) # dataset label
    new_line.append(scene_name)   # subscene label
    new_line.append(frame) #frame
    new_line.append(id_) #id

    new_line.append(x) #x
    new_line.append(y) #y
    new_line.append(undefined) # xmin. The top left x-coordinate of the bounding box.
    new_line.append(undefined) # ymin The top left y-coordinate of the bounding box.
    new_line.append(undefined) # xmax. The bottom right x-coordinate of the bounding box.
    new_line.append(undefined) # ymax. The bottom right y-coordinate of the bounding box.

    new_line.append(undefined) # label type of agent    

    return new_line

def transpose(theta,points,x,y):
    R = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ]
    T = [x,y]
    new_points = []

    for point in points:
        new_point = np.add(np.matmul(R,point), T)
        new_points.append(new_point)
    return np.array(new_points)



def get_bbox(theta,width,length,x,y):
    t,b = [],[]

    points = [
        [-length,width/2.0],
        [-length,-width/2.0],
        [0,width/2.0],
        [0,-width/2.0], 
        [-length/2.0,0],         
    ]

    new_points = transpose(theta,points,x,y)
    

    

    max_int = 10e30
    t = [0,-max_int]
    b = [0,max_int]
    pm = new_points[-1]
    for p in new_points[:-1]:
        if p[0] < pm[0] and p[1] > t[1]:
            t = p
        if p[0] > pm[0] and p[1] < b[1]:
            b = p
    
    

    # return new_points
    return t,b,pm


"""
    Input: Standardized file_path
    Output a dictionnary of trajectories:
    {
        traj_id: {
            coordinates: [],
            bboxes: [],
            frames: [],
            scene:

        }
    }
"""

def extract_trajectories(file_name,destination_path = "", save = False):

    trajectories = {}

    with open(file_name) as file_:
        file_ = csv.reader(file_, delimiter=',')
        for line in file_:

            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])

            if id_ not in trajectories:

                trajectories[id_] = {
                    "coordinates" : [],
                    "bboxes" : [],
                    "frames" : [],
                    "scene" : line[1],
                    "user_type" : line[10],
                    "id" : id_,
                    "dataset" : line[0]
                }
            trajectories[id_]["coordinates"].append(coordinates)
            trajectories[id_]["bboxes"].append(bbox)
            trajectories[id_]["frames"].append(frame)

    if save:

        if os.path.exists(destination_path):
            os.remove(destination_path)
        with open(destination_path,"a") as scene_txt:
            for key in trajectories:
                line = trajectories[key]
                # trajectories["id"] = key
                line = json.dumps(line)
                # print(line)
                # print("------")
                scene_txt.write(line + "\n" )
    else:
        return trajectories
    return

"""
    Input: Standardized file_path
    Output a dictionnary of frames:
    {
        frame: {
            object_id : {
                coordinates : [],
                bbox : []
            }

        }
    }
"""
import json

def extract_frames(file_path,destination_path = "", save = False):
    frames = {}

    

    with open(file_path) as file_:
        for line in file_:
            line = line.split(",")
            
            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])
            type_ = line[10]
            

            if frame not in frames:
                frames[frame] = {}
    
            frames[frame][id_] = {
                "coordinates" : coordinates,
                "bbox" : bbox,
                "type" : type_,
                "scene" : line[1],
                "dataset" : line[0]

                }
            if save:
                frames[frame]["frame"] = frame


        if save:

            if os.path.exists(destination_path):
                os.remove(destination_path)
            with open(destination_path,"a") as scene_txt:
                for key in sorted(frames):
                    line = frames[key]
                    # line["frame"] = key
                    line = json.dumps(line)
                    # print(line)
                    # print("------")
                    scene_txt.write(line + "\n" )
        else:
            return frames
    return