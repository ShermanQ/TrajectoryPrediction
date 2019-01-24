import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories

import seaborn as sns

import time


import helpers_a
import json
import csv
import matplotlib.pyplot as plt


ROOT = "./../"
CSV = ROOT + "extractors/csv/"

import pandas as pd



from matplotlib.lines import Line2D      
from matplotlib.patches import Polygon
def plot_street(param,depth):
    lines = [
        plt.Polygon([[-param,-param],[0,-param],[0,0],[-param,0]],color = "black",fill = False), #5
        plt.Polygon([[0,0],[param,0],[param,-param],[0,-param]],color = "black",fill = False),#7
        plt.Polygon([[0,0],[param,0],[param,param],[0,param]],color = "black",fill = False),#8
        plt.Polygon([[0,0],[0,param],[-param,param],[-param,0]],color = "black",fill = False),#6
        plt.Polygon([[-depth,-param],[-param,-param],[-param,0],[-depth,0]],color = "black",fill = False,closed = False),#4
        plt.Polygon([[-depth,0],[-param,0],[-param,param],[-depth,param]],color = "black",fill = False,closed = False),#12
        plt.Polygon([[-param,-depth],[-param,-param],[0,-param],[0,-depth]],color = "black",fill = False,closed = False),#9
        plt.Polygon([[0,-depth],[0,-param],[param,-param],[param,-depth]],color = "black",fill = False,closed = False),#1
        plt.Polygon([[depth,param],[param,param],[param,0],[depth,0]],color = "black",fill = False,closed = False),#3
        plt.Polygon([[depth,0],[param,0],[param,-param],[depth,-param]],color = "black",fill = False,closed = False),#10
        plt.Polygon([[-param,depth],[-param,param],[0,param],[0,depth]],color = "black",fill = False,closed = False),#2
        plt.Polygon([[0,depth],[0,param],[param,param],[param,depth]],color = "black",fill = False,closed = False) #11
    ]
    for line in lines:
        plt.gca().add_line(line)
 
    

def plot_trajectories(file_path, user_type = None,temp_path = "./temp.txt"):
    extract_trajectories(file_path,destination_path = temp_path, save = True)

    param = 5
    depth = 20

    #5
    
    plot_street(param,depth)
    with open(temp_path) as trajectories:
        # ids = [4,5]
        # ids = [0,1,2,3,4,5,6]

        for i,trajectory in enumerate(trajectories):
            trajectory = json.loads(trajectory)
            coordinates = trajectory["coordinates"]
            id_ = trajectory["id"]
            # if id_ in ids:
            id_scene = 4
            if trajectory["scene"] == "minute number: "+str(id_scene):
                if user_type == None:
                    x = [p[0] for p in coordinates]
                    y = [p[1] for p in coordinates]

                    
                    plt.plot(x,y)
                    # plt.show() 

                elif trajectory["user_type"] == user_type:

                
                    x = [p[0] for p in coordinates]
                    y = [p[1] for p in coordinates]

                    
                    plt.plot(x,y)
        plt.show()    
        
        # print(i)
        os.remove(temp_path)
        
    return

def main():

    csv_file = "new_rates/bad_30.0to2.5.csv"
    csv_file = "bad.csv"
    file_path = CSV + csv_file
    user = "car\n"
    # user = "pedestrian\n"

    plot_trajectories(file_path, user_type = user)


if __name__ == "__main__":
    main()