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

NB_FRAME = 1000


PRINT_EVERY = 300

FRAMERATE = 10

def add_traj(trajectories,row,current_trajectories,trajectory_counter):
    current_trajectories.append(trajectory_counter)
    trajectories[trajectory_counter] = {
        "coordinates":[[float(row[0]),float(row[1])]],
        "frames":[int(row[4])],
        "type": row[2]
        }
    trajectory_counter += 1
    # return trajectories,current_trajectories,trajectory_counter
    return trajectory_counter


def angle(p0,p1):
    diff = np.subtract(p1,p0)
    return np.arctan2(diff[1],diff[0])
def old_angle(coordinates):
    if len(coordinates) < 2:
        return np.random.uniform(low = -1.57,high = 1.57)
    else:
        return angle(coordinates[-2],coordinates[-1])


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

def persist_trajectories(outdated_trajectories,trajectories,writer):
    for c in outdated_trajectories:
        type_ = trajectories[c]["type"]
        trajectory_id = c 
        unknown = -1
        
        for point,bframe in zip(trajectories[c]["coordinates"],trajectories[c]["frames"]):
            new_line = []
            new_line.append(DATASET) # dataset label
            new_line.append(DATASET) # subscene label
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
import matplotlib.pyplot as plt

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



    # covariance matrix for x,y,theta,deltat(s)
    cov = [0.005,0.005,1.57/10., 5.0/FRAMERATE ]
    cov = [0.005,0.005,3.0/FRAMERATE]
    cov = [0.005,0.005]

    # the consecutive number of not updated frame for a trajectory
    inactivity = 150

    probability_threshold = 0.

    ctr,ctr1,ctr2,i = 0,0,0,0 # proba eliminated # no trajectory left # different type
 
    # bad_csv = "/home/laurent/Documents/master/extractors/csv/bad.csv"
    with open(bad_csv,"a+") as csv_file:
        writer = csv.writer(csv_file)

        with open(TRAJECTORIES_PATH) as csv_file:
        # with open("/home/laurent/Documents/master/extractors/datasets/bad/trajectories.csv") as csv_file:

            traj_reader = csv.reader(csv_file, delimiter=',')
            for i,row in enumerate(traj_reader):

                if i% 50000 == 0:
                    print(i,ctr,ctr1,ctr2,time.time()-s)

                x1,y1,init,frame1,type1 = float(row[0]),float(row[1]),row[3],int(row[4]),row[2]
            

                # update current_trajectories by removing the old trajectories
                outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame1)
                # when a trajectory is outdated, write its content in destination file and delete it from the memory
                persist_trajectories(outdated_trajectories,trajectories,writer)
                

                # if no trajectories available, the observation necessarily belongs to a new one
                # this will be removed when the adding of a new trajectory will only depend on the true value on the row

                # if len(current_trajectories) == 0:
                ## add new trajectory only if init is true
                if init == "True":           
                    trajectory_counter = add_traj(trajectories,row,current_trajectories,trajectory_counter)
                elif len(current_trajectories) == 0:
                    ctr1 +=1
                else:
                    # to get rid of the problem: manuel label late in comparison to detection
                    if len(current_trajectories) != 0:
                        # initialize probabilities to 0
                        probabilities = np.zeros(len(current_trajectories))
                        prior = 1./float(len(current_trajectories))
                        xs = []
                        types = []
                        # for each currently available trajectory
                        for k,c in enumerate(current_trajectories):

                            # get selected trajectory data
                            x0,y0 = trajectories[c]["coordinates"][-1][0],trajectories[c]["coordinates"][-1][1]
                            type0,frame0 = trajectories[c]["type"],trajectories[c]["frames"][-1]
                            # if the frame of the new row is anterior to the last frame of the selected trajectory, the belonging proba stays 0
                            # if frame0 < frame1:
                            theta0 = old_angle(trajectories[c]["coordinates"])
                            theta1 = angle([x0,y0],[x1,y1])
                            t0 = float(frame0)/float(FRAMERATE)
                            t1 = float(frame1)/float(FRAMERATE)

                            # mean = [x0,y0,theta0,t0]
                            # x = [x1,y1,theta1,t1]

                            # mean = [x0,y0,1./FRAMERATE]
                            # x = [x1,y1,t1-t0]
                            mean = [x0,y0]
                            x = [x1,y1]
                            xs.append(mean)
                            types.append(type0)
                            
                            # compute for the selected trajectory the probability of belonging
                            probabilities[k] = compute_map(prior,x, mean, cov )
                            # stats.multivariate_normal.pdf(x,mean=mean,cov=cov)
                                
                                

                       # normalize maps to get probability
                        # if np.sum(probabilities) > 0:
                        if len(current_trajectories) > 1:
                            probabilities /= np.sum(probabilities)
                        # get the index of the most likely trajectory
                        choice = best_trajectory(current_trajectories,probabilities,probability_threshold)

                        if choice != -1 :
                            
                            if types[np.argmax(probabilities)] == type1:
                                
                                trajectories[choice]["coordinates"].append([x1,y1])
                                trajectories[choice]["frames"].append(frame1)
                            else:
                                ctr2 += 1

                        else:
                            ctr += 1

        persist_trajectories(current_trajectories,trajectories,writer)
    
    print(ctr,i)


    print(time.time()-s)
    
if __name__ == "__main__":
    main()
