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

FRAMERATE = 30

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
    cov = [0.001,0.001,1.57, 1.0/FRAMERATE ]

    # the consecutive number of not updated frame for a trajectory
    inactivity = 50

    probability_threshold = 0.2

    with open(TRAJECTORIES_PATH) as csv_file:
        traj_reader = csv.reader(csv_file, delimiter=',')
        for i,row in enumerate(traj_reader):

            if i% 100000 == 0:
                print(i)
                print(len(trajectories.keys()))
                print(time.time()-s)

            x1,y1,init,frame1 = float(row[0]),float(row[1]),row[3],int(row[4])
          

            # update current_trajectories by removing the old trajectories
            outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame1)
            for c in outdated_trajectories:
                del trajectories[c]
            

            # if no trajectories available, the observation necessarily belongs to a new one
            # this will be removed when the adding of a new trajectory will only depend on the true value on the row

            # if len(current_trajectories) == 0:
            ## add new trajectory only if init is true
            if init == "True":           
                trajectory_counter = add_traj(trajectories,row,current_trajectories,trajectory_counter)

            else:
                # to get rid of the problem: manuel label late in comparison to detection
                if len(current_trajectories) != 0:
                    # initialize probabilities to 0
                    probabilities = np.zeros(len(current_trajectories))
                    
                    # for each currently available trajectory
                    for k,c in enumerate(current_trajectories):

                        # get selected trajectory data
                        x0,y0,frame0 = trajectories[c]["coordinates"][-1][0],trajectories[c]["coordinates"][-1][1],trajectories[c]["frames"][-1]

                        # if the frame of the new row is anterior to the last frame of the selected trajectory, the belonging proba stays 0
                        if frame0 < frame1:
                            theta0 = old_angle(trajectories[c]["coordinates"])
                            theta1 = angle([x0,y0],[x1,y1])
                            t0 = float(frame0)/float(FRAMERATE)
                            t1 = float(frame1)/float(FRAMERATE)

                            mean = [x0,y0,theta0,t0]
                            x = [x1,y1,theta1,t1]

                            # compute for the selected trajectory the probability of belonging
                            probabilities[k] = stats.multivariate_normal.pdf(x,mean=mean,cov=cov) # this is not between 0 and 1
                    # print(probabilities)
                    # get the index of the most likely trajectory
                    choice = best_trajectory(current_trajectories,probabilities,probability_threshold)
                    # print(choice)
                    if choice != -1:
                        trajectories[choice]["coordinates"].append([x1,y1])
                        trajectories[choice]["frames"].append(frame1)
                    # if no trajectory was selected, add a new one
                    # this will be removed when the adding of a new trajectory will only depend on the true value on the row
                    # else:
                        
                    #     trajectory_counter = add_traj(trajectories,row,current_trajectories,trajectory_counter)

                    # print("nb traj: " + str(len(trajectories.keys())))



    print(time.time()-s)
if __name__ == "__main__":
    main()
