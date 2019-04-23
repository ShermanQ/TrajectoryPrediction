import sys
import json
import csv
import numpy as np
import pandas as pd
import helpers
from scipy.spatial.distance import euclidean
import os

class StopsRemover():
    def __init__(self,preprocessing_params,data,prepare_params):
        preprocessing_params = json.load(open(preprocessing_params)) 
        prepare_params = json.load(open(prepare_params)) 
        data = json.load(open(data))


        # self.scenes = preprocessing_params["scenes"]
        self.original_file = data["filtered_datasets"] + "{}.csv"
        # self.types = preprocessing_params["types"]
        self.trajectories_temp = data["temp"] + "trajectories.txt"
        # self.scaler_dest = data["scalers"] 
        self.temp = data["temp"] + "temp.csv"
        self.dist_threshold = preprocessing_params["distance_threshold"]
        # self.delta_t = 1.0/float(prepare_params["framerate"])
        # self.acc_thresh = preprocessing_params["acceleration_threshold"]
        # self.dec_thresh = preprocessing_params["deceleration_threshold"]

        
    def remove_stopped(self,scene):
        helpers.remove_file(self.temp)
        os.rename(self.original_file.format(scene),self.temp)
        helpers.remove_file(self.original_file.format(scene))

        helpers.extract_trajectories(self.temp,self.trajectories_temp,save=True)
        ctr = 0
        with open(self.trajectories_temp) as trajectories:
            with open(self.original_file.format(scene),"a+") as f:
                csv_writer = csv.writer(f)

                for k,trajectory in enumerate(trajectories):                
                    trajectory = json.loads(trajectory)
                    coordinates = trajectory["coordinates"]
                    start_point = coordinates[0]
                    stop_point = coordinates[-1]
                    distance = euclidean(start_point,stop_point)
                    if distance < self.dist_threshold:
                        ctr += 1
                    else:
                        rows = helpers.save_traj(trajectory)
                        for row in rows:
                            csv_writer.writerow(row)

        # os.rename(self.temp,self.original_file.format(scene))
        helpers.remove_file(self.temp)
        
        return (ctr,k+1)



        
# python preprocess_datasets/stops_remover.py parameters/preprocessing.json parameters/data.json
def main():
    args = sys.argv
    stops = StopsRemover(args[1],args[2],args[3])
    # stats.get_stats()



if __name__ == "__main__":
    main()