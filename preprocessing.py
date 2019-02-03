import extractors.helpers as helpers
import json
import matplotlib.pyplot as plt
import os 
from scipy.spatial import distance

def plot_coordinates(coordinates):
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    plt.plot(x,y)  

def plot_points(coordinates):
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    plt.scatter(x,y) 

def display_trajectory(trajectory):
    coordinates = trajectory["coordinates"]
    plot_coordinates(coordinates)
    plot_points(coordinates)
   

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

def find_outlier_points(trajectory,framerate = 0.1,acceleration_thresh = 5, deceleration_thresh = -8):
    coordinates = trajectory["coordinates"]
    speeds = get_speeds(coordinates,framerate)
    speeds = [speeds[0]] + speeds 
    accelerations = get_accelerations(speeds,framerate)

    counter = 0
    indices = []
    outlier_accelerations = []
    for i,a in enumerate(accelerations):
        if not( a > deceleration_thresh and a < acceleration_thresh):
            outlier_accelerations.append(a)
            counter += 1
            if i not in indices:
                indices.append(i)
            if i + 1 not in indices:
                indices.append(i+1)
    outlier_points = [coordinates[i] for i in indices]
    

    return counter,outlier_points,outlier_accelerations
    
 # # # #  # # # # # #
  # # # #  # # # # #
def main():
    filepath = "./data/csv/lankershim_inter2.csv"
    temp_path = "./data/temp/temp.txt"
    helpers.extract_trajectories(filepath,destination_path = temp_path, save = True)

    try:
        with open(temp_path) as trajectories:
            for i,trajectory in enumerate(trajectories):
                trajectory = json.loads(trajectory)

                nb_outliers,outlier_points,outlier_accelerations = find_outlier_points(trajectory)
                if nb_outliers > 0:
                    print(i)
                    print(nb_outliers)
                    print(outlier_accelerations)
                    display_trajectory(trajectory)
                    plot_points(outlier_points)
                    plt.show()
    except:
        os.remove(temp_path)  
if __name__ == "__main__":
    main()