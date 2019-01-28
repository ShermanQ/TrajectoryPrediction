import os
import csv
import time
import numpy as np
from scipy import stats


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
    lane = start_lane([row[0],row[1]])
    trajectories[trajectory_counter] = {
        "coordinates":[[row[0],row[1]]],
        "frames":[row[4]],
        "type": row[2],
        "angles": [angle],
        "subscene" : subscene ,
        "lane" : lane                                                  #############                 
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

def start_lane(point,origin = [0.,-1.5],boundary = 5.):
    x = point[0]
    y = point[1]
    if x < origin[0]-boundary:
        return 4
    elif x > origin[0] + boundary:
        return 2
    elif y < origin[1] -boundary:
        return 1
    elif y > origin[1] + boundary:
        return 3
    elif x > origin[0]:
        if y < origin[1]:
            return 10
        else:
            return 11
    else:
        if y < origin[1]:
            return 9
        else:
            return 12
def grid(value,test_values,return_values):
    if value < test_values[0]:
        return return_values[0]
    elif value < test_values[1]:
        return return_values[1]
    elif value < test_values[2]:
        return return_values[2]
    else:
        return return_values[3] 

def current_lane(point,origin = [0.,-1.5],boundary = 5.):
    x = point[0]
    y = point[1]
    ytest_values = [origin[1]-boundary,origin[1],origin[1] + boundary]
    if x < origin[0]-boundary:
        return_values = [16,4,8,15]
        return grid(y,ytest_values,return_values)
    elif x < origin[0]:
        return_values = [5,9,12,3]
        return grid(y,ytest_values,return_values)
    elif x < origin[0] + boundary:

        return_values = [1,10,11,7]
        return grid(y,ytest_values,return_values)
    else:
        return_values = [13,6,2,14]
        return grid(y,ytest_values,return_values)



     


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
    map_ = stats.multivariate_normal.pdf(x,mean=mean,cov=cov,allow_singular=True) * prior
    
    return map_

def origin_angle(point):
    ux = [1,0]
    point /= np.linalg.norm(point)
    prod = np.dot(ux,point)
    angle = np.arccos(prod)
    # if point[1] < 0:
    #     angle *= -1.0
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

def add_frame_car(rows,trajectories,current_trajectories,cov,counter,forbidden_states,available_transitions,probability_threshold = 10e-50,start_cst = False,trans_cst = False,sc = 10., tc = 10.):
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
        xlane = current_lane(x) ######### ########### ########
        ax = origin_angle(x)
        x.append(ax)
        ###########################################################################################
        if start_cst:
            xpenalty1 = 0.
            x.append(xpenalty1)

        if trans_cst:
            xpenalty2 = 0.
            x.append(xpenalty2)
        ###########################################################################################
        for j,key1 in enumerate(available_trajectories):
            trajectory = available_trajectories[key1]
            am = trajectory["angles"][-1]
            mean = copy.copy(trajectory["coordinates"][-1])
            tlane = trajectory["lane"]
            mlane = current_lane(mean)
            # if tlane == 1:
            #     print(2)
            mean.append(am)
            ######################################################################################
            if start_cst:
                mean.append(0.)
            if trans_cst:
                mean.append(0.)
            
            if start_cst and trans_cst:
                if xlane in forbidden_states[tlane]:
                    x[-2] = sc
                if xlane not in available_transitions[mlane]:
                    x[-1] = tc
            elif start_cst:
                if xlane in forbidden_states[tlane]:
                    x[-1] = sc
            elif trans_cst:
                if xlane not in available_transitions[mlane]:
                    x[-1] = tc
            #######################################################################################
            proba = compute_map(prior,x, mean, cov )
            if xlane in available_transitions[mlane]:
                probas[i][j] = proba
    nb_points = len(available_points.keys())

    while nb_points > 0:
        idx = np.unravel_index(probas.argmax(), probas.shape)
        i = [key for key in available_points.keys()][idx[0]]
        j = [key for key in available_trajectories.keys()][idx[1]]

        row = available_points[i]
        x = [row[0],row[1]]
        ax = origin_angle(x)

        if probas.max() > probability_threshold:
            trajectories[j]["coordinates"].append([row[0],row[1]])
            trajectories[j]["frames"].append(row[4])
            trajectories[j]["angles"].append(ax)
        probas[idx[0],:] = -1
        probas[:,idx[1]] = -1
        nb_points -=1

    return counter

def add_frame_pedestrian(rows,trajectories,current_trajectories,cov,counter,probability_threshold = 10e-50):
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

        if probas.max() > probability_threshold:
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
    desired_angle =  0.34
    var = (desired_width/2.0) ** 2
    var_angle = (desired_angle/2.0) ** 2
    cov = [var,var]
    cov_ped = [var,var,var_angle]

    forbidden_states = {
        1:[3,2,15,16],
        2:[1,3,4,6,7,13],
        3:[1,2,7,16],
        4:[1,2,3,15],
        9:[1,2,3,4,8,15],
        10:[1,2,3,4,5,15,16],
        11:[1,2,3,4,6,13],
        12:[1,2,3,4,7,16]
        
    }
    available_transitions = {
        1:[1,6,10,13],
        2:[2,7,11,14],
        3:[3,8,12,15],
        4:[4,5,9,16],
        5:[5],
        6:[6],
        7:[7],
        8:[8],
        9:[5,9,10,11,12],
        10:[10,6,11,12],
        11:[11,14,7,10,12,9],
        12:[12,8,9,10,11],
        13: [13,6],
        14: [14,7],
        15: [15,8],
        16: [16,5],
        
    }
    start_cst = True
    trans_cst = False
    sc = 10. 
    tc = 20.
    if start_cst and trans_cst:
        cov = [var,var,var_angle,1,1]
    elif start_cst:
        cov = [var,var,var_angle,1]
    elif trans_cst:
        cov = [var,var,var_angle,1]
    else:
        cov = [var,var,var_angle]



    # the consecutive number of not updated frame for a trajectory
    inactivity = 100
    probability_threshold = 10e-30
    

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
                trajectory_counter = add_frame_car(car_rows,trajectories,car_current_trajectories,cov,trajectory_counter,forbidden_states,available_transitions,probability_threshold,start_cst,trans_cst,sc, tc )
                # print(4544)
                trajectory_counter = add_frame_pedestrian(ped_rows,trajectories,ped_current_trajectories,cov_ped,trajectory_counter,probability_threshold)

                # in case where new trajectories were added
                current_trajectories = car_current_trajectories + ped_current_trajectories
                
        persist_trajectories(current_trajectories,trajectories,writer)
 
if __name__ == "__main__":
    main()
