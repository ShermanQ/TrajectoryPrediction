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





def main():

    csv_file = "new_rates/bad_30.0to2.5.csv"
    # csv_file = "koper.csv"
    file_path = CSV + csv_file
    temp_path = "./temp.txt"
    extract_trajectories(file_path,destination_path = temp_path, save = True)

    with open(temp_path) as trajectories:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i,trajectory in enumerate(trajectories):
            trajectory = json.loads(trajectory)
            coordinates = trajectory["coordinates"]
            # print(coordinates)
            # print("----")
            # print(coordinates)
            # print(trajectory["user_type"])
            if trajectory["user_type"] == "car\n":

               
                x = [p[0] for p in coordinates]
                y = [p[1] for p in coordinates]

                
                if i < 20:
                    plt.plot(x,y)
                    

                
                # sns.lineplot(x = "x", y = "y", data = pd.DataFrame(coordinates, columns = ["x","y"]), sort= False)
            
            
        plt.show()    
        
    print(i)
    os.remove(temp_path)

if __name__ == "__main__":
    main()