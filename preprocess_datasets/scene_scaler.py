import os 
import csv
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import helpers
import json
import sys
import joblib
import json
import time

class ScalersComputer():

    def __init__(self,data,scene_list,scale = True):
        data = json.load(open(data))

        # self.center = center
        self.temp = data["temp"] + "temp.csv"
        self.original_file = data["filtered_datasets"] + "{}.csv"
        self.scaler_dest = data["scalers"] 
        self.scene_list = scene_list
        self.scale = scale
        self.trajectories_temp = data["temp"] + "trajectories.txt"
        
        # if scale:
        #     print("loading scaler")
        #     self.__get_scaler()
        #     print("done!")

    def __get_scaler(self):
        # mms = MinMaxScaler(feature_range=(0,1))

        min_ = 1e30
        max_ = -1e30
        print("Computing normalization!")
        for scene in self.scene_list:
            # print(scene)

            min_x,max_x,min_y,max_y = self.__get_boudaries(self.original_file.format(scene))
            
          

            min_scene = min(min_x ,min_y ) 
            max_scene = max(max_x ,max_y )


            min_ = min(min_scene,min_)
            max_ = max(max_scene,max_)
        print(min_,max_)

        # mms = mms.fit([[min_],[max_]])
        # self.scaler = mms

        # helpers.remove_file(self.scaler_dest)
        # joblib.dump(self.scaler, self.scaler_dest) 
        print("Done!")
        return min_,max_




    

        
    def __get_boudaries(self,file_path):
        with open(file_path) as scene_csv:
            data_reader = csv.reader(scene_csv)

            min_x,min_y = 10e30,10e30
            max_x,max_y = 10e-30,10e-30
            
            for row in data_reader:
          
                x = np.min([[float(row[i])] for i in range(4,10,2) if float(row[i]) != -10000])
                y = np.min([[float(row[i])] for i in range(5,11,2) if float(row[i]) != -10000])

                if x < min_x and x != -1:
                    min_x = x
                if y < min_y and y != -1:
                    min_y = y
                if x > max_x and x != -1:
                    max_x = x
                if y > max_y and y != -1:
                    max_y = y
        return min_x,max_x,min_y,max_y

    def get_offset_scalers(self):

        s = time.time()
        min_,max_ = self.__get_scaler()

        print("Computing means")
        
        means = {}
        nb_offsets = 0
        for scene in self.scene_list:
            # print(scene)
            
            offsetsx_sum = 0
            offsetsy_sum = 0

            helpers.extract_trajectories(self.original_file.format(scene),self.trajectories_temp,save = True)
            with open(self.trajectories_temp) as trajectories:
                for k,trajectory in enumerate(trajectories):
                    trajectory = json.loads(trajectory)
                    coordinates = np.array(trajectory["coordinates"])
                    x = coordinates[:,0]
                    y = coordinates[:,1]
                    offsets_x = helpers.get_offsets(x)
                    offsets_y = helpers.get_offsets(y)

                    nb_offsets += len(x)
                    offsetsx_sum += np.sum(offsets_x)
                    offsetsy_sum += np.sum(offsets_y)

            mean_x = offsetsx_sum 
            mean_y = offsetsy_sum 

            means[scene] = {
                "x":mean_x,
                "y":mean_y
            }

        mean_x = 0. 
        mean_y = 0. 

        for i,scene in enumerate(means):
            mean_x += means[scene]["x"]
            mean_y += means[scene]["y"]

        mean_x /= nb_offsets
        mean_y /= nb_offsets

        print("x_mean,y_mean")
        print(mean_x,mean_y)

        print("Done!")

        print("Computing standard deviation")
        nb_offsets = 0
        vars = {}
        for scene in self.scene_list:
            # print(scene)
            offsetsx_sum = 0
            offsetsy_sum = 0

            helpers.extract_trajectories(self.original_file.format(scene),self.trajectories_temp,save = True)
            with open(self.trajectories_temp) as trajectories:
                for k,trajectory in enumerate(trajectories):
                    trajectory = json.loads(trajectory)
                    coordinates = np.array(trajectory["coordinates"])
                    x = coordinates[:,0]
                    y = coordinates[:,1]
                    offsets_x = helpers.get_offsets(x)
                    offsets_y = helpers.get_offsets(y)

                    offsets_x -= mean_x
                    offsets_y -= mean_y 

                    offsets_x = np.square(offsets_x)
                    offsets_y = np.square(offsets_y)


                    nb_offsets += len(x)
                    offsetsx_sum += np.sum(offsets_x)
                    offsetsy_sum += np.sum(offsets_y)

            var_x = offsetsx_sum 
            var_y = offsetsy_sum 

            vars[scene] = {
                "x":var_x,
                "y":var_y
            }


        var_x = 0. 
        var_y = 0. 

        for i,scene in enumerate(vars):
            var_x += vars[scene]["x"]
            var_y += vars[scene]["y"]

        var_x /= nb_offsets
        var_y /= nb_offsets

        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)

        print("var_x,var_y")
        print(var_x,var_y)

        print("std_x,std_y")
        print(std_x,std_y)
        print("Done!")

        mean_x = int(10000*mean_x)/1000.0
        mean_y = int(10000*mean_y)/1000.0
        std_x = int(10000*std_x)/1000.0
        std_y = int(10000*std_y)/1000.0

        scalers = {

            "standardization":{"meanx":mean_x,"meany":mean_y,"stdx":std_x,"stdy":std_y},
            "normalization":{"min":min_,"max":max_}
        }

        helpers.remove_file(self.scaler_dest)
        json.dump(scalers, open(self.scaler_dest,"w") )

        print(time.time()-s)

def main():

    args = sys.argv

    # scene_scaler = SceneScaler(args[1],int(args[2]))
    # scene_scaler.min_max_scale(args[3])

    



        


        





                    
if __name__ == "__main__":
    main()