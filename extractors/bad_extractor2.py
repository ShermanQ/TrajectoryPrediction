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


def add_traj(trajectories,row,current_trajectories,trajectory_counter):
    current_trajectories.append(trajectory_counter)
    angle = origin_angle([row[0],row[1]])

    trajectories[trajectory_counter] = {
        "coordinates":[[row[0],row[1]]],
        "frames":[row[4]],
        "type": row[2],
        "angles": [angle]
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
def add_frame(rows,trajectories,current_trajectories,cov,counter):
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
    probas = np.zeros((len(available_points),len(current_trajectories)))
    prior = 1./ len(current_trajectories)
    
    
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
            row = [float(row[0]),float(row[1]),row[2],row[3],int(row[4])]

            frame0 = row[4]
            frame = frame0
            while frame < nb_frames:
                print(frame)
                if frame == 245:
                    print("dg")
                # update current_trajectories by removing the old trajectories
                outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame)
                # when a trajectory is outdated, write its content in destination file and delete it from the memory
                persist_trajectories(outdated_trajectories,trajectories,writer)
                
                rows = [row]
                
                try:
                    while frame == frame0:
                        row = next(traj_reader)
                        row = [float(row[0]),float(row[1]),row[2],row[3],int(row[4])]
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
                trajectory_counter = add_frame(car_rows,trajectories,car_current_trajectories,cov,trajectory_counter)
                # trajectory_counter = add_frame(ped_rows,trajectories,ped_current_trajectories,cov,trajectory_counter)

                # in case where new trajectories were added
                current_trajectories = car_current_trajectories + ped_current_trajectories
                
        persist_trajectories(current_trajectories,trajectories,writer)
   

    #         for i,row in enumerate(traj_reader):
    #             if i == 167:
    #                 print(167)
    #             if i% 100 == 0:
    #                 print(i,ctr,ctr1,ctr2,time.time()-s)

    #             p1,init,frame1,type1 = [float(row[0]),float(row[1])],row[3],int(row[4]),row[2]
    #             # p1 = transformer.inverse(p1)[0]
    #             x1,y1 = p1[0],p1[1]

                
                
    #             row = [x1,y1,type1,init,frame1]
    #             # update current_trajectories by removing the old trajectories
    #             outdated_trajectories,current_trajectories = old_trajectories(current_trajectories,trajectories,inactivity,frame1)
    #             # when a trajectory is outdated, write its content in destination file and delete it from the memory
    #             persist_trajectories(outdated_trajectories,trajectories,writer)
                

    #             correct_trajectories = []
    #             lane1 = get_current_lane(p1)

    #             for cur in current_trajectories:
    #                 # if trajectories[cur]["type"] == type1 and trajectories[cur]["frames"][-1] < frame1:
    #                 dist = np.linalg.norm(np.subtract(p1,trajectories[cur]["coordinates"][-1]))
    #                 start_lane0 = trajectories[cur]["start_lane"]
    #                 if trajectories[cur]["type"] == type1:
    #                     if trajectories[cur]["frames"][-1] <= frame1:
    #                         if dist < distance_threshold:
    #                             if type1 == "pedestrian" or (lane1 in start_transitions[start_lane0] ):
    #                                 correct_trajectories.append(cur)
                
     
    #             ## add new trajectory only if init is true
    #             if init == "True":       
    #                 starting_lane = get_starting_lane(p1) 
    #                 if type1 == "pedestrian":
    #                        starting_lane = -1
    #                 trajectory_counter = add_traj(trajectories,row,current_trajectories,trajectory_counter,starting_lane)
    #             elif len(correct_trajectories) == 0:

    #                 ctr1 +=1
    #             else:
    #                 # to get rid of the problem: manuel label late in comparison to detection
    #                 if len(correct_trajectories) != 0:
                    
    #                     probabilities = np.zeros(len(correct_trajectories))
    #                     prior = 1./float(len(correct_trajectories))
                        
    #                     xs = []
    #                     types = []

                        
    #                     # for each currently available trajectory
    #                     for k,c in enumerate(correct_trajectories):


    #                         # get selected trajectory data
    #                         x0,y0 = trajectories[c]["coordinates"][-1][0],trajectories[c]["coordinates"][-1][1]
    #                         type0,frame0 = trajectories[c]["type"],trajectories[c]["frames"][-1]
                            
                            
    #                         # if the frame of the new row is anterior to the last frame of the selected trajectory, the belonging proba stays 0
               
    #                         # t0 = float(frame0)/float(FRAMERATE)
    #                         # t1 = float(frame1)/float(FRAMERATE)

    #                         # angle = reel_angle(trajectories[c]["coordinates"],[x1,y1])
    #                         angle = old_angle(trajectories[c]["coordinates"])
    #                         n_angle =  old_angle([trajectories[c]["coordinates"][-1],[x1,y1]])
                            
                        
    #                         mean = [x0,y0]
    #                         x = [x1,y1]
    #                         xs.append(mean)
    #                         types.append(type0)
                            
    #                         # compute for the selected trajectory the probability of belonging
    #                         probabilities[k] = compute_map(prior,x, mean, cov )
    #                         # probabilities[k] = measure_probability(cov, mean, x) * prior
                                
                                

    #                    # normalize maps to get probability
    #                     if np.sum(probabilities) > 0:
                            
    #                         probabilities /= np.sum(probabilities)
                        
                            
    #                     # get the index of the most likely trajectory
    #                     choice = best_trajectory(correct_trajectories,probabilities,probability_threshold)


    #                     if choice != -1 :
                           
                            
    #                         if frame1 > trajectories[choice]["frames"][-1] :
                                
    #                             trajectories[choice]["coordinates"].append([x1,y1])
    #                             trajectories[choice]["frames"].append(frame1)
    #                         else:
    #                             ctr2 += 1

    #                     else:
    #                         ctr += 1
           

    #     persist_trajectories(current_trajectories,trajectories,writer)
    
    # print(ctr,i)


    # print(time.time()-s)
    
if __name__ == "__main__":
    main()
