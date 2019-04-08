import sys
import json
import csv
import numpy as np
import pandas as pd
import helpers
from scipy.spatial.distance import euclidean
from preprocess_datasets.preprocessing import get_speeds,get_accelerations,nb_outlier_points

class Stats():
    def __init__(self,preprocessing_params,data,prepare_params):
        preprocessing_params = json.load(open(preprocessing_params)) 
        prepare_params = json.load(open(prepare_params)) 

        data = json.load(open(data))
        self.scenes = preprocessing_params["scenes"]
        self.extracted_datasets = data["filtered_datasets"] + "{}.csv"
        self.types = preprocessing_params["types"]
        self.trajectories_temp = data["temp"] + "trajectories.txt"
        self.scaler_dest = data["scalers"] 
        self.dist_threshold = preprocessing_params["distance_threshold"]
        self.delta_t = 1.0/float(prepare_params["framerate"])
        self.acc_thresh = preprocessing_params["acceleration_threshold"]
        self.dec_thresh = preprocessing_params["deceleration_threshold"]

        


    def get_stats(self):
        tot1 = 0
        tot2 = 0
        for scene in self.scenes:
            print(scene)
            # nbs = self.__get_nbs(scene)
            # print(nbs)

            # stats = self.__get_frames_stats(scene)
            # print(stats)

            # ctr = self.__get_stopped(scene)
            # print(ctr)
            # tot1 += ctr[0]
            # tot2 += ctr[1]

            # ctr = self.__get_outlier_accelerations(scene)
            # print(ctr)
            # tot1 += ctr[0]
            # tot2 += ctr[1]

        print(tot1,tot2)

    def __get_nbs(self,scene):

        ids = []
        nbs = {"total":0}

        with open(self.extracted_datasets.format(scene)) as scene_file:
            scene_reader = csv.reader(scene_file)
            for row in scene_reader:
                id_ = row[3]
                type_ = row[-1]

                if id_ not in ids:
                    if type_ not in nbs:
                        nbs[type_] = 0
                    nbs[type_] += 1
                    nbs["total"]+= 1                    
                    ids.append(id_)
        return nbs

    def __get_stopped(self,scene):
        helpers.extract_trajectories(self.extracted_datasets.format(scene),self.trajectories_temp,save=True)
        ctr = 0
        with open(self.trajectories_temp) as trajectories:
            for k,trajectory in enumerate(trajectories):                
                trajectory = json.loads(trajectory)
                coordinates = trajectory["coordinates"]
                start_point = coordinates[0]
                stop_point = coordinates[-1]
                distance = euclidean(start_point,stop_point)
                if distance < self.dist_threshold:
                    ctr += 1
        return (ctr,k+1)

    def __get_outlier_accelerations(self,scene):
        helpers.extract_trajectories(self.extracted_datasets.format(scene),self.trajectories_temp,save=True)
        ctr = 0
        nbs = []
        with open(self.trajectories_temp) as trajectories:
            for k,trajectory in enumerate(trajectories):                
                trajectory = json.loads(trajectory)
                nb_out = nb_outlier_points(trajectory,self.delta_t,self.acc_thresh,self.dec_thresh)
                if nb_out > 0:
                    ctr += 1
                    nbs.append(nb_out)
        
        return (ctr,k+1,np.sum(nbs),np.mean(nbs))
        

        

    def __get_frames_stats(self,scene):

        frames = {}
        with open(self.extracted_datasets.format(scene)) as scene_file:
            scene_reader = csv.reader(scene_file)
            for row in scene_reader:
                frame = row[2]
                type_ = row[-1]

                if frame not in frames:
                    frames[frame] = {
                        "total":0
                    }

                if type_ not in frames[frame]:
                    frames[frame][type_] = 0


                frames[frame][type_] += 1
                frames[frame]["total"] += 1
        stats = np.zeros( (len(frames.keys()),len(self.types ) ) )

        # print(stats.shape)
        for i,key in enumerate(sorted(frames)):
            for j,key1 in enumerate((self.types)):
                if key1 in frames[key]:
                    stats[i,j] = frames[key][key1]

        s = pd.DataFrame(stats,columns = self.types)
        stats = s.describe().drop("count")
        
        return stats


                


        
# python preprocess_datasets/stats.py parameters/preprocessing.json parameters/data.json
def main():
    args = sys.argv
    stats = Stats(args[1],args[2])
    stats.get_stats()



if __name__ == "__main__":
    main()