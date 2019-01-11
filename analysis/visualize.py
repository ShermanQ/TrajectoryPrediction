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



def plot_trajectories(file_path, user_type = None,temp_path = "./temp.txt"):
    extract_trajectories(file_path,destination_path = temp_path, save = True)

    with open(temp_path) as trajectories:
        for i,trajectory in enumerate(trajectories):
            trajectory = json.loads(trajectory)
            coordinates = trajectory["coordinates"]
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
    # csv_file = "fsc_6.csv"
    file_path = CSV + csv_file

    plot_trajectories(file_path, user_type = "car\n")


if __name__ == "__main__":
    main()