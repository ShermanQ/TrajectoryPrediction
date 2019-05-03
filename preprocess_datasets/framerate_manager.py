import sys 
import os
import helpers
import json
import csv

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import splev, splrep


class FramerateManager():
    def __init__(self,data,new_rate):
        data = json.load(open(data))

        self.framerates_json = json.load(open(data["framerates_json"]))
        self.new_rate = new_rate
        self.temp = data["temp"] + "{}.txt"
        self.destination_file = data["filtered_datasets"] + "{}.csv"
        self.original_file = data["extracted_datasets"] + "{}.csv"


    def change_rate(self,scene_name):
        
        # print("worker {} starting".format(scene_name))
        former_rate = float(self.framerates_json[scene_name])
        
        rate_ratio = int(former_rate/self.new_rate)

        # self.destination_file = self.destination_file.format(scene_name)
        
        helpers.remove_file(self.destination_file.format(scene_name))

        self.counter = 0

        helpers.extract_trajectories(self.original_file.format(scene_name),self.temp.format(scene_name),save=True)
        with open(self.temp.format(scene_name)) as trajectories:

            for k,trajectory in enumerate(trajectories):     
                              
                    trajectory = json.loads(trajectory)
                    coordinates = np.array(trajectory["coordinates"])

                    #processing 
                    x = coordinates[:,0]
                    y = coordinates[:,1]

                    nb_sample = len(x)
                    t = np.array([1/former_rate*i for i in range(nb_sample)])

                    nb_sample1 = int(nb_sample/rate_ratio) + 1 ######
                    t1 = np.array([1/self.new_rate*i for i in range(nb_sample1)])

                    sx = splrep(t, x, s = 0)
                    sy = splrep(t, y, s = 0)

                    x_down = splev(t1, sx)
                    y_down = splev(t1, sy)

                    coordinates_down = np.concatenate([x_down[:,np.newaxis],y_down[:,np.newaxis]],axis = 1).tolist()

                    # plt.plot(x,y,"ro")
                    # plt.plot(x_down,y_down,"bx")
                    # plt.show()

                    trajectory["coordinates"] = coordinates_down










                    #
                    self.__write_trajectory(trajectory,scene_name)
           
        helpers.remove_file(self.temp.format(scene_name)) 
        # print("worker {} exiting".format(scene_name))

        

    def __write_trajectory(self,trajectory,scene):
        with open(self.destination_file.format(scene),"a") as file_:
            file_writer = csv.writer(file_)
            rows = []
            coordinates = trajectory["coordinates"]
            bboxes = trajectory["bboxes"]
            frames = trajectory["frames"]

            scene = trajectory["scene"]
            user_type = trajectory["user_type"]
            dataset = trajectory["dataset"]
            id_ = trajectory["id"]


            traj_len = len(coordinates)

            scenes = [scene for _ in range(traj_len)]
            user_types = [user_type for _ in range(traj_len)]
            datasets = [dataset for _ in range(traj_len)]
            ids = [id_ for _ in range(traj_len)]


            for d,s,f,i,c,b,t in zip(datasets,scenes,frames,ids,coordinates,bboxes,user_types):
                row = []
                row.append(d)
                row.append(s)
                row.append(f)
                row.append(i)
                for e in c:
                    row.append(e)
                for e in b:
                    row.append(e)
                row.append(t)

                file_writer.writerow(row)

            
            
            

        
    # def __write_frame(self,frame,scene):

    #     with open(self.destination_file.format(scene),"a") as file_:
    #         file_writer = csv.writer(file_)
    #         for key in frame:
    #             new_line = []
    #             new_line.append(frame[key]["dataset"])
    #             new_line.append(frame[key]["scene"])
    #             new_line.append(self.counter)
    #             new_line.append(key)
    #             for e in frame[key]["coordinates"]:
    #                 new_line.append(e)

    #             for e in frame[key]["bbox"]:
    #                 new_line.append(e)

    #             new_line.append(frame[key]["type"].split("\n")[0])


    #             file_writer.writerow(new_line)



    # def change_rate(self,scene_name):
        
    #     # print("worker {} starting".format(scene_name))
    #     former_rate = float(self.framerates_json[scene_name])
        
    #     rate_ratio = int(former_rate/self.new_rate)

    #     # self.destination_file = self.destination_file.format(scene_name)
        
    #     helpers.remove_file(self.destination_file.format(scene_name))

    #     self.counter = 0

    #     helpers.extract_frames(self.original_file.format(scene_name),self.temp.format(scene_name),save=True)
    #     with open(self.temp.format(scene_name)) as frames:
    #         for frame in frames:
    #             frame = json.loads(frame)
    #             i = frame["frame"]
    #             if i % rate_ratio == 0:
    #                 self.__write_frame(frame["ids"],scene_name)
    #                 self.counter += 1   
    #     helpers.remove_file(self.temp.format(scene_name)) 
    #     # print("worker {} exiting".format(scene_name))

#python prepare_training/framerate_manager.py parameters/data.json 2.5 lankershim_inter2
def main():

    args = sys.argv
    framerate_manager = FramerateManager(args[1],float(args[2]))

    framerate_manager.change_rate(args[3])
  
if __name__ == "__main__":
    main()