import extractors.helpers as helpers
import json
import matplotlib.pyplot as plt
import os 
from scipy.spatial import distance

def plot_coordinates(coordinates):
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    plt.plot(x,y)   

def display_trajectory(trajectory):
    coordinates = trajectory["coordinates"]
    plot_coordinates(coordinates)
    plt.show()

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

def find_outlier_points(trajectory,framerate = 0.1,acceleration_thresh = 5, deceleration_thresh = -8):
    coordinates = trajectory["coordinates"]
    speeds = get_speeds(coordinates,framerate)
    accelerations = get_speeds(coordinates,framerate)

    counter = 0
    for a in accelerations:
        if not( a > deceleration_thresh and a < acceleration_thresh):
            counter += 1
    return counter
    
def main():
    filepath = "./data/csv/lankershim_inter2.csv"
    temp_path = "./data/temp/temp.txt"
    helpers.extract_trajectories(filepath,destination_path = temp_path, save = True)

    try:
        with open(temp_path) as trajectories:
            for i,trajectory in enumerate(trajectories):
                trajectory = json.loads(trajectory)

                nb_outliers = find_outlier_points(trajectory)
                if nb_outliers > 0:
                    print(i)
                    print(nb_outliers)

                    display_trajectory(trajectory)
    except:
        os.remove(temp_path)  
if __name__ == "__main__":
    main()