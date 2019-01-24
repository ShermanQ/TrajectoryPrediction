import os
import helpers
from helpers import bb_intersection_over_union as iou
import csv
import pandas as pd
import time
import itertools
import numpy as np
from helpers import remove_char,parse_init_line,parse_boxes_line

import scipy.stats as stats

ROOT = "./datasets/"
DATASET = "bad"
DATA = ROOT + DATASET + "/"
BOXES_PATH = DATA + "boxes.csv"
INIT_PATH = DATA + "init.csv"
TRAJECTORIES_PATH = DATA + "trajectories.csv"
dict_classes = {'7': 'car', '15':'pedestrian', '6': 'car', '14': 'pedestrian'}
CSV_PATH = "./csv/"
HOMOGRAPHY = "datasets/bad/homography/"


def add_traj(trajectories,row,current_trajectories,trajectory_counter): ###########
    current_trajectories.append(trajectory_counter)
    angle = origin_angle([row[0],row[1]])
    subscene = row[5]
    trajectories[trajectory_counter] = {
        "coordinates":[[row[0],row[1]]],
        "frames":[row[4]],
        "type": row[2],
        "angles": [angle],
        "subscene" : subscene                                                   #############                 
        }
    trajectory_counter += 1
    # return trajectories,current_trajectories,trajectory_counter
    return trajectory_counter

def best_trajectory(current_trajectories,probabilities,probability_threshold):
    best_proba = np.max(probabilities)
    if best_proba > probability_threshold:
        return current_trajectories[np.argmax(probabilities)]
    else:
        return -1
def old_trajectories(current_trajectories,trajectories,inactivity,current_frame):
    ids = []
    new_current = []
    for c in current_trajectories:
        trajectory = trajectories[c]
        if abs(current_frame - trajectory["frames"][-1] ) > inactivity:
            ids.append(c)
        else:
            new_current.append(c)
    return ids,new_current
def linear_interpolation(coordinates,frames):
    new_coordinates = []
    new_frames = []

    for i in range(1,len(frames)):
        point = [coordinates[i-1][0],coordinates[i-1][1]]
        frame = frames[i-1]
        new_coordinates.append(point)
        new_frames.append(frame)

        if frames[i-1] + 1 < frames[i] :
            offset = frames[i] - frames[i-1] -1
            x0 = coordinates[i-1][0]
            y0 = coordinates[i-1][1]
            x1 = coordinates[i][0]
            y1 = coordinates[i][1]
            dx = (x1 - x0)/float(offset)
            dy = (y1 - y0)/float(offset)
            for j in range(offset):
                new_point = [x0 + (j+1) * dx, y0 + (j+1) * dy ]
                new_frame = frames[i-1] + j + 1
                new_coordinates.append(new_point)
                new_frames.append(new_frame)
    return new_coordinates,new_frames

def persist_trajectories(outdated_trajectories,trajectories,writer,interpolate = True):
    for c in outdated_trajectories:
        type_ = trajectories[c]["type"]
        subscene = trajectories[c]["subscene"]                      ######################
        trajectory_id = c 
        unknown = -1
        
        coordinates = trajectories[c]["coordinates"]
        frames = trajectories[c]["frames"]

        if interpolate:
            coordinates,frames = linear_interpolation(coordinates,frames)
                

        for point,bframe in zip(coordinates,frames):
            new_line = []
            new_line.append(DATASET) # dataset label
            new_line.append(subscene) # subscene label                       ###########################
            new_line.append(bframe) #frame
            new_line.append(trajectory_id) #id
            new_line.append(point[0]) #x
            new_line.append(point[1]) #y
            new_line.append(unknown)# xmin. The top left x-coordinate of the bounding box.
            new_line.append(unknown)# ymin The top left y-coordinate of the bounding box.
            new_line.append(unknown)# xmax. The bottom right x-coordinate of the bounding box
            new_line.append(unknown)# ymax. The bottom right y-coordinate of the bounding box
            new_line.append(type_) # label type of agent   
            writer.writerow(new_line)

        del trajectories[c]
def compute_map(prior,x, mean, cov ):
    map_ = stats.multivariate_normal.pdf(x,mean=mean,cov=cov) * prior
    
    return map_

def origin_angle(point):
    ux = [1,0]
    point /= np.linalg.norm(point)
    prod = np.dot(ux,point)
    angle = np.arccos(prod)
    return angle

def get_nb_frames(filepath):
    nb = 0
    with open(filepath) as csv_file:
        traj_reader = csv.reader(csv_file, delimiter=',')
        start_frame = 0
        frame = 0
        for i,row in enumerate(traj_reader):
            frame = int(row[4])
            if i ==0:
                start_frame = frame
        nb = frame - start_frame + 1
    return nb
import copy
def add_frame(rows,trajectories,current_trajectories,cov,counter,distance_threshold):
    available_points = {}
    available_trajectories = {}

    for i, row in enumerate(rows):
        available_points[i] = row
    for c in current_trajectories:
        available_trajectories[c] = trajectories[c]
    
    delete_rows = []
    for key in available_points:
        row = available_points[key]
        if row[3] == "True":
            counter = add_traj(trajectories,row,current_trajectories,counter)
            delete_rows.append(key)
    for key in delete_rows:
        del available_points[key]


    # if no point left stop
    if len(available_points.keys()) == 0 :
        return counter
    # if no trajectory left stop
    if len(available_trajectories.keys()) == 0 :
        return counter

    # i: points_id  j: trajectories ids
    probas = np.zeros((len(available_points.keys()),len(available_trajectories.keys())))
    prior = 1./ len(available_trajectories.keys())    #prior = 1./ len(available_points) #
    
    
    for i,key in enumerate(available_points):
        row = available_points[key]
        x = [row[0],row[1]]
        ax = origin_angle(x)
        x.append(ax)

        for j,key1 in enumerate(available_trajectories):
            trajectory = available_trajectories[key1]
            am = trajectory["angles"][-1]
            mean = copy.copy(trajectory["coordinates"][-1])
            mean.append(am)

            proba = compute_map(prior,x, mean, cov )
            
            probas[i][j] = proba
    nb_points = len(available_points.keys())
    while nb_points > 0:
        idx = np.unravel_index(probas.argmax(), probas.shape)
        i = [key for key in available_points.keys()][idx[0]]
        j = [key for key in available_trajectories.keys()][idx[1]]

        row = available_points[i]
        x = [row[0],row[1]]
        ax = origin_angle(x)

        dist = np.linalg.norm(np.subtract(x,trajectories[j]["coordinates"][-1]))
        if dist < distance_threshold:
            trajectories[j]["coordinates"].append([row[0],row[1]])
            trajectories[j]["frames"].append(row[4])
            trajectories[j]["angles"].append(ax)
        probas[idx[0],:] = -1
        probas[:,idx[1]] = -1
        nb_points -=1

    return counter



def main():


    s = time.time()

    bad_csv = CSV_PATH + DATASET + ".csv"
    if os.path.exists(bad_csv):
        os.remove(bad_csv)

    # dict containing the trajectories and their coordinates
    trajectories = {}

    # counter to assign a new trajectory its id
    trajectory_counter = 0

    # ids of the available trajectories
    current_trajectories = []

   
    #####
    # covariance matrix for x,y,theta,deltat(s)
    # height,width = 720, 1280
    desired_width = 2.0 # meters
    desired_angle =  0.17
    var = (desired_width/2.0) ** 2
    var_angle = (desired_angle/2.0) ** 2
    distance_threshold = 15.0
    cov = [var,var]
    cov = [var,var,var_angle]


    # the consecutive number of not updated frame for a trajectory
    inactivity = 100

    

    bad_csv = "/home/laurent/Documents/master/extractors/csv/bad.csv"
    TRAJECTORIES_PATH = "/home/laurent/Documents/master/extractors/datasets/bad/trajectories.csv"

    nb_frames = get_nb_frames(TRAJECTORIES_PATH)
    
    
    with open(bad_csv,"a+") as csv_file:
        writer = csv.writer(csv_file)        
        with open(TRAJECTORIES_PATH) as csv_file:

            traj_reader = csv.reader(csv_file, delimiter=',')

            row = next(traj_reader)
            row = [float(row[0]),float(row[1]),row[2],row[3],int(row[4]),row[5]]

            frame0 = row[4]
            frame = frame0
            while frame < nb_frames:
                if frame % 50000 == 0:
                    print(frame)
                    print(time.time() - s)
                    
                # update current_trajectories by removing the old trajectories
                outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame)
                # when a trajectory is outdated, write its content in destination file and delete it from the memory
                persist_trajectories(outdated_trajectories,trajectories,writer)
                
                rows = [row]
                
                try:
                    while frame == frame0:
                        row = next(traj_reader)
                        row = [float(row[0]),float(row[1]),row[2],row[3],int(row[4]),row[5]]
                        frame = row[4]
                        if frame == frame0:
                            rows.append(row)
                    frame0 = frame
                    
                except:
                    frame += 1
                # separate pedestrians and cars rows
                car_rows = [row for row in rows if row[2] == "car"]
                ped_rows = [row for row in rows if row[2] == "pedestrian"]
                # separate pedestrians and cars current_trajectories
                car_current_trajectories = [id_ for id_ in current_trajectories if trajectories[id_]["type"] == "car"]
                ped_current_trajectories = [id_ for id_ in current_trajectories if trajectories[id_]["type"] == "pedestrian"]
                # add frames for pedestrians and cars respectively
                trajectory_counter = add_frame(car_rows,trajectories,car_current_trajectories,cov,trajectory_counter,distance_threshold)
                trajectory_counter = add_frame(ped_rows,trajectories,ped_current_trajectories,cov,trajectory_counter,distance_threshold)

                # in case where new trajectories were added
                current_trajectories = car_current_trajectories + ped_current_trajectories
                
        persist_trajectories(current_trajectories,trajectories,writer)
 
if __name__ == "__main__":
    main()
