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



NB_FRAME = 1000


PRINT_EVERY = 300

FRAMERATE = 10

# def add_traj(trajectories,row,current_trajectories,trajectory_counter):
#     current_trajectories.append(trajectory_counter)
#     trajectories[trajectory_counter] = {
#         "coordinates":[[float(row[0]),float(row[1])]],
#         "frames":[int(row[4])],
#         "type": row[2]
#         }
#     trajectory_counter += 1
#     # return trajectories,current_trajectories,trajectory_counter
#     return trajectory_counter

def add_traj(trajectories,row,current_trajectories,trajectory_counter,starting_lane):
    current_trajectories.append(trajectory_counter)
    trajectories[trajectory_counter] = {
        "coordinates":[[row[0],row[1]]],
        "frames":[row[4]],
        "type": row[2],
        "start_lane":starting_lane
        }
    trajectory_counter += 1
    # return trajectories,current_trajectories,trajectory_counter
    return trajectory_counter


def angle(p0,p1):
    diff = np.subtract(p1,p0)
    return np.arctan2(diff[1],diff[0])
def old_angle(coordinates):
    if len(coordinates) < 2:
        return 0.
    else:
        return angle(coordinates[-2],coordinates[-1])

# def reel_angle(coordinates,v1):
#     if len(coordinates) < 2:
#         return 0.
#     else:
#         return angle1(np.subtract(coordinates[-1],coordinates[-2]),np.subtract(v1,coordinates[-1]))
# def angle1(v0,v1):
#     v0 /= np.linalg.norm(v0)
#     v1 /= np.linalg.norm(v1)

#     angle = np.arccos(np.dot(v0,v1))
#     return angle

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
    map_ = np.exp(stats.multivariate_normal.logpdf(x,mean=mean,cov=cov)) * prior
    
    return map_


def euclidean_angle_distance(angle_1,angle_2):

    return np.abs((angle_1+np.pi - angle_2)%(2*np.pi)-np.pi)


import math
def measure_probability(cov, mean, state):


    x_delta = mean[0] - state[0]
    y_delta = mean[1] - state[1]
    # angle_delta = euclidean_angle_distance(mean[2], state[2])
    
    # lane_validity_delta = mean[3] - state[3]
    cov = np.linalg.inv(np.diag(cov))
    normalization_cst = np.linalg.det(2*math.pi* cov) ** (-0.5)

    # dif_state = [x_delta, y_delta, angle_delta]
    dif_state = [x_delta, y_delta]

    dt = np.dot(cov, dif_state)

    prob = np.exp(-1.*np.dot(dif_state, dt ))

    prob /= normalization_cst
    
    
    return prob

# def measure_probability(cov, mean, state):


#     x_delta = mean[0] - state[0]
#     y_delta = mean[1] - state[1]
#     angle_delta = euclidean_angle_distance(mean[2], state[2])
    
#     # lane_validity_delta = mean[3] - state[3]
#     cov = np.diag(cov)

#     dif_state = [x_delta, y_delta, angle_delta]
#     dt = np.dot(cov, dif_state)

#     prob = -1.*np.dot(dif_state, dt )

#     # prob /= normalization_cst
    
    
#     return prob
        # 2 # 1  #
        #   #    #
    ####################
    # 3 # 6 #  8 # 3
    ####################
    # 4 # 5 #  7 # 4
    ####################
        # 2 # 1  #
        #   #    #
def get_starting_lane(start_point):
    x = start_point[0]
    y = start_point[1]
    if in_the_middle(x,y):
        return get_middle_quadrant(start_point)
    else:
        if y < -5.:
            return 1
        elif y > 5.:
            return 2
        elif x > 5.:
            return 3
        elif x < -5.:
            return 4
def in_the_middle(x,y):
    return (x > -5 and x < 5) and (y > -5 and y < 5)

def get_middle_quadrant(point):
    x = point[0]
    y = point [1]
    if in_the_middle(x,y):
        if x < 0:
            if y < 0:
                return 5
            else:
                return 6
        elif x > 0:
            if y < 0:
                return 7
            else:
                return 8

def get_current_lane(current_point):
    x = current_point[0]
    y = current_point[1]

    if in_the_middle(x,y):
        return get_middle_quadrant(current_point)
    else:
        if x > 0 and x < 5:
            if y > 0:
                return 11
            else:
                return 1
        elif x > -5 and x < 0:
            if y > 0:
                return 2
            else:
                return 9
            
        elif y > 0 and y < 5:
            if x > 0:
                return 3
            else:
                return 12
        elif y > -5 and y < 0:
            if x > 0:
                return 10
            else:
                return 4
  

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

    available_transitions = {
        1 : [7,1],
        2 : [6,2],
        3 : [8,3],
        4 : [5,4],
        5 : [9,7,5],
        6 : [12,5,6],
        7 : [10,8,7],
        8 : [11,6,8],
        9: [9],
        10: [10],
        11: [11],
        12: [12]
    }
    

    # key is the starting lane, list is authorized lanes considering the starting one 
    start_transitions = {
        1 : [7,8,6,3,4,11,1],
        2 : [6,3,5,7,4,9,2],
        3 : [8,11,6,5,9,12,3],
        4 : [5,9,7,8,11,10,4],
        5 : [9,7,10,8,11,5],
        6 : [12,5,9,7,10,6],
        7 : [8,6,12,11,10,7],
        8 : [11,6,12,5,9,8],
        9: [9],
        10: [10],
        11: [11],
        12: [12]
    }

    # covariance matrix for x,y,theta,deltat(s)
    cov = [0.005,0.005,1.57/10., 5.0/FRAMERATE ]
    cov = [0.005,0.005,3.0/FRAMERATE]
    # height,width = 720, 1280
    cov = [0.5,0.5]

    # the consecutive number of not updated frame for a trajectory
    inactivity = 100

    probability_threshold = 0.
    distance_threshold = 50.0

    ctr,ctr1,ctr2,i = 0,0,0,0 # proba eliminated # no trajectory left # different type
 
    bad_csv = "/home/laurent/Documents/master/extractors/csv/bad.csv"
    with open(bad_csv,"a+") as csv_file:
        writer = csv.writer(csv_file)

        TRAJECTORIES_PATH = "/home/laurent/Documents/master/extractors/datasets/bad/trajectories.csv"
        with open(TRAJECTORIES_PATH) as csv_file:

            traj_reader = csv.reader(csv_file, delimiter=',')
            for i,row in enumerate(traj_reader):
                if i == 167:
                    print(167)
                if i% 100 == 0:
                    print(i,ctr,ctr1,ctr2,time.time()-s)

                p1,init,frame1,type1 = [float(row[0]),float(row[1])],row[3],int(row[4]),row[2]
                # p1 = transformer.inverse(p1)[0]
                x1,y1 = p1[0],p1[1]

                
                
                row = [x1,y1,type1,init,frame1]
                # update current_trajectories by removing the old trajectories
                outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame1)
                # when a trajectory is outdated, write its content in destination file and delete it from the memory
                persist_trajectories(outdated_trajectories,trajectories,writer)
                

                correct_trajectories = []
                lane1 = get_current_lane(p1)

                for cur in current_trajectories:
                    # if trajectories[cur]["type"] == type1 and trajectories[cur]["frames"][-1] < frame1:
                    dist = np.linalg.norm(np.subtract(p1,trajectories[cur]["coordinates"][-1]))
                    start_lane0 = trajectories[cur]["start_lane"]
                    if trajectories[cur]["type"] == type1:
                        if trajectories[cur]["frames"][-1] <= frame1:
                            if dist < distance_threshold:
                                if type1 == "pedestrian" or (lane1 in start_transitions[start_lane0] and lane1 in available_transitions[start_lane0]):
                                    correct_trajectories.append(cur)
                
     
                ## add new trajectory only if init is true
                if init == "True":       
                    starting_lane = get_starting_lane(p1) 
                    if type1 == "pedestrian":
                           starting_lane = -1
                    trajectory_counter = add_traj(trajectories,row,current_trajectories,trajectory_counter,starting_lane)
                elif len(correct_trajectories) == 0:

                    ctr1 +=1
                else:
                    # to get rid of the problem: manuel label late in comparison to detection
                    if len(correct_trajectories) != 0:
                    
                        probabilities = np.zeros(len(correct_trajectories))
                        prior = 1./float(len(correct_trajectories))
                        
                        xs = []
                        types = []

                        
                        # for each currently available trajectory
                        for k,c in enumerate(correct_trajectories):


                            # get selected trajectory data
                            x0,y0 = trajectories[c]["coordinates"][-1][0],trajectories[c]["coordinates"][-1][1]
                            type0,frame0 = trajectories[c]["type"],trajectories[c]["frames"][-1]
                            
                            
                            # if the frame of the new row is anterior to the last frame of the selected trajectory, the belonging proba stays 0
               
                            # t0 = float(frame0)/float(FRAMERATE)
                            # t1 = float(frame1)/float(FRAMERATE)

                            # angle = reel_angle(trajectories[c]["coordinates"],[x1,y1])
                            angle = old_angle(trajectories[c]["coordinates"])
                            n_angle =  old_angle([trajectories[c]["coordinates"][-1],[x1,y1]])
                            
                        
                            mean = [x0,y0]
                            x = [x1,y1]
                            xs.append(mean)
                            types.append(type0)
                            
                            # compute for the selected trajectory the probability of belonging
                            probabilities[k] = compute_map(prior,x, mean, cov )
                            # probabilities[k] = measure_probability(cov, mean, x) * prior
                                
                                

                       # normalize maps to get probability
                        if np.sum(probabilities) > 0:
                            
                            probabilities /= np.sum(probabilities)
                        
                            
                        # get the index of the most likely trajectory
                        choice = best_trajectory(correct_trajectories,probabilities,probability_threshold)


                        if choice != -1 :
                           
                            
                            if frame1 > trajectories[choice]["frames"][-1] :
                                
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
