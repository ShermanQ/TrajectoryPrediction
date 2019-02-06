import extractors.helpers as helpers
import json
import matplotlib.pyplot as plt
import os 
from scipy.spatial import distance
from scipy.interpolate import CubicSpline
import numpy as np
import csv
from scipy.signal import butter,sosfilt,filtfilt

def plot_coordinates(coordinates,subplot):
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    subplot.plot(x,y)  

def plot_points(coordinates,subplot):
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    subplot.scatter(x,y) 

def display_trajectory(trajectory,subplot):
    coordinates = trajectory["coordinates"]
    plot_coordinates(coordinates,subplot)
    plot_points(coordinates,subplot)
   

def get_speed(point1,point2,deltat):
    d = distance.euclidean(point1,point2)
    v = d/deltat
    return v
def get_speeds(coordinates,framerate):
    speeds = []
    for i in range(1,len(coordinates)):
        speed = get_speed(coordinates[i-1],coordinates[i],framerate)
        speeds.append(speed)
    return speeds

def get_acceleration(v1,v2,deltat):
    a = (v2-v1)/deltat
    return a

def get_accelerations(speeds,framerate):
    accelerations = []
    for i in range(1,len(speeds)):
        acceleration = get_acceleration(speeds[i-1],speeds[i],framerate)
        accelerations.append(acceleration)
    return accelerations

# def find_outlier_points(trajectory,framerate = 0.1,acceleration_thresh = 5, deceleration_thresh = -8):
#     coordinates = trajectory["coordinates"]
#     speeds = get_speeds(coordinates,framerate)
#     # speeds = [speeds[0]] + speeds 
#     speeds = speeds 

#     accelerations = get_accelerations(speeds,framerate)

#     counter = 0
#     indices = []
#     outlier_accelerations = []
#     for i,a in enumerate(accelerations):
#         if not( a > deceleration_thresh and a < acceleration_thresh):
#             outlier_accelerations.append(a)
#             counter += 1
#             if i+1 not in indices:
#                 indices.append(i+1)
#             if i + 2 not in indices:
#                 indices.append(i+2)
#     outlier_points = [coordinates[i] for i in indices]
    

#     return counter,outlier_points,outlier_accelerations,indices
# def interpolate(trajectory,indices):
#     coordinates = trajectory["coordinates"]
#     ps = []
#     t = []
#     for i,c in enumerate(coordinates):
#         if i not in indices:
#             ps.append(c)
#             t.append(i*0.1)


#     y1 = [c[0] for c in ps]
#     y2 = [c[1] for c in ps]

#     cs1 = CubicSpline(t, y1)
#     cs2 = CubicSpline(t, y2)

#     ts = [i*0.1 for i in range(len(coordinates))]

#     x = cs1(ts)
#     y = cs2(ts)

#     coord = []
#     for a,b in zip(x,y):
#         coord.append([a,b])

#     return {"coordinates" : coord}

def indices_to_sequence(indices):
    if len(indices) == 0:
        return []
    indices_sequences = [[indices[0]]]
    for i in range(1,len(indices)):
        if indices[i] == indices[i-1] + 1:
            indices_sequences[-1].append(indices[i])
        else:
            indices_sequences.append([indices[i]])
    return indices_sequences

def find_outlier_points(trajectory,framerate = 0.1,acceleration_thresh = 5, deceleration_thresh = -8):
    coordinates = trajectory["coordinates"]
    speeds = get_speeds(coordinates,framerate)
    # speeds = [speeds[0]] + speeds 
    speeds = speeds 

    accelerations = get_accelerations(speeds,framerate)

    
    indices = []
    for i,a in enumerate(accelerations):
        if not( a > deceleration_thresh and a < acceleration_thresh):
            if i+1 not in indices:
                indices.append(i+1)
            if i + 2 not in indices:
                indices.append(i+2)

    indices_sequences = indices_to_sequence(indices)
    
    outlier_points = [coordinates[i] for i in indices]

    return indices_sequences,outlier_points,accelerations,speeds


def interpolate(trajectory,indices):
    coordinates = trajectory["coordinates"]

    nb_reference = 10
    nb_points = len(coordinates)
    delete_indices = []
    for s in indices:
        if s[0]  >= nb_reference and nb_points - s[-1]  >= nb_reference:
            ps = []
            t = []

            for i in range(s[0] - nb_reference,s[-1] + nb_reference):
                
                if i not in s:
                    t.append(i*0.1)
                    ps.append(coordinates[i])
            
            y1 = [c[0] for c in ps]
            y2 = [c[1] for c in ps]

            cs1 = CubicSpline(t, y1)
            cs2 = CubicSpline(t, y2)

            ts = [i*0.1 for i in s]
            # ts = [i*0.1 for i in range(s[0] - nb_reference,s[-1] + nb_reference)]


            x = cs1(ts)
            y = cs2(ts)

            coord_s = []
            for a,b in zip(x,y):
                coord_s.append([a,b])
            for i,c in zip(s,coord_s):
                coordinates[i] = c

    #     else: 
    #         if s[0]  < nb_reference:
    #             for i in range(s[-1]):
    #                 delete_indices.append(i)
    #         if nb_points - s[-1]  < nb_reference:
    #             for i in range(s[0],nb_points):
    #                 delete_indices.append(i)
    # new_coordinates = []
    # for i,c in enumerate(coordinates):
    #     if i not in delete_indices:
    #         new_coordinates.append(c)

    # return {"coordinates" : new_coordinates}
    return {"coordinates" : coordinates}

def apply_filter(speeds):
    # sampling frequency fs is 10 Hz or s-1,
    # niquist frequency is fn = f/2.0 = 5.0 Hz
    # Paper says cut-off frequency of 1 Hz which is 0.2 * fn
    b,a = butter(1,0.8,fs = 10 ) 

    filtered_speeds = filtfilt(b,a, speeds)
    return filtered_speeds

import math
def new_pos(new_speeds,coordinates,dt = 0.1):
    coords = [coordinates[0]]
    for i in range(1,len(coordinates)):
        if i == 181:
            print(i)
        new_speed = new_speeds[i-1]
        pos = coordinates[i]
        old_pos = coords[-1]
        dir_vec = np.subtract(pos,old_pos)
        new_pos = old_pos
        if np.linalg.norm(dir_vec) > 0:
            dir_vec /= np.linalg.norm(dir_vec)
            theta = np.arccos(np.dot(dir_vec,[1,0]))
            if dir_vec[1] < 0:
                theta = 2.* math.pi - theta
           
            new_x = old_pos[0] + new_speed * np.cos(theta) * dt
            new_y = old_pos[1] + new_speed * np.sin(theta) *dt
            new_pos = [new_x, new_y]

        coords.append(new_pos)
    return {"coordinates": coords}


def main():
    filepath = "./data/csv/lankershim_inter2.csv"
    temp_path = "./data/temp/temp.txt"
    helpers.extract_trajectories(filepath,destination_path = temp_path, save = True)


    # with open(save_path) as save_csv:
        # csv_writer =csv.writer(save_csv)
    with open(temp_path) as trajectories:
        for i,trajectory in enumerate(trajectories):
            trajectory = json.loads(trajectory)

            # nb_outliers,outlier_points,outlier_accelerations,indices = find_outlier_points(trajectory)
            
            indices,outlier_points,accelerations1,speeds1 = find_outlier_points(trajectory)
            
            
            nb_outliers = np.sum([len(t) for t in indices])
            if nb_outliers > 0:
                print(i,nb_outliers)
                print(indices)
                f, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3,2)

                print(i)
                print(nb_outliers)
                # print(outlier_accelerations)
                display_trajectory(trajectory,ax1)
                plot_points(outlier_points,ax1)
                
            

                # print(indices)
                traj = interpolate(trajectory,indices)
                indices,outlier_points,accelerations2,speeds2 = find_outlier_points(traj)
                filtered_speeds = apply_filter(speeds2)
                filtered_accelerations = get_accelerations(filtered_speeds,0.1)
                new_traj = new_pos(filtered_speeds,traj["coordinates"])

                display_trajectory(traj,ax3)
                plot_points(outlier_points,ax3)
                print(accelerations2)
                ax2.plot(speeds1)
                ax2.plot(speeds2)
                ax2.plot(filtered_speeds)

                ax4.plot(accelerations1)
                ax4.plot(accelerations2)
                ax4.plot(filtered_accelerations)

                display_trajectory(new_traj,ax5)
                nspeeds = get_speeds(new_traj["coordinates"],0.1)
                nacc = get_accelerations(nspeeds,0.1)
                print(nspeeds)
                ax6.plot(nspeeds)
                ax6.plot(nacc)
                plt.show()

                

     
if __name__ == "__main__":
    main()