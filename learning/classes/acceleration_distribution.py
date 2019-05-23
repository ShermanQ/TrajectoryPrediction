from helpers import extract_trajectories,get_speeds,get_accelerations,remove_file
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Accelerations():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))

        self.scenes = self.prepare_params["selected_scenes"]
        # self.scenes = self.prepare_params["toy_scenes"]

        self.scene_paths = self.data_params["filtered_datasets"] + "{}.csv"
        self.report_path = self.data_params["dynamics"]
        self.trajectories_temp = self.data_params["temp"] + "trajectories.txt"
        self.delta_t = 1.0/float(self.prepare_params["framerate"])

        print(self.delta_t)


    def get_distribs(self):
        types_dict = {}
        trajectory_len = []
        ctr = 0
        for scene in self.scenes:
            print(scene)
            remove_file(self.trajectories_temp)
            extract_trajectories(self.scene_paths.format(scene),self.trajectories_temp,save=True)

            scene_accelerations = []

            with open(self.trajectories_temp) as trajectories:

                for k,trajectory in enumerate(trajectories):     
                    ctr += 1           
                    trajectory = json.loads(trajectory)
                    coordinates = trajectory["coordinates"]
                    type_ = trajectory["user_type"]
                    if type_ not in types_dict:
                        types_dict[type_] = {
                            "accelerations":[],
                            "speeds":[]
                        }
                    
                    trajectory_len.append(len(coordinates))
                    speeds = get_speeds(coordinates,self.delta_t)
                    accelerations = get_accelerations(speeds,self.delta_t)
                    
                    types_dict[type_]["accelerations"].append(accelerations)
                    types_dict[type_]["speeds"].append(speeds)

        
        for key in types_dict:
            types_dict[key]["accelerations"] = np.concatenate(types_dict[key]["accelerations"])
            types_dict[key]["speeds"] = np.concatenate(types_dict[key]["speeds"])

        thresholds = {}
        for key in types_dict:
            thresholds[key] = {
                "speeds": {"mean": np.mean(types_dict[key]["speeds"]), "std": np.std(types_dict[key]["speeds"])},
                "accelerations": {"mean": np.mean(types_dict[key]["accelerations"]), "std": np.std(types_dict[key]["accelerations"])}
            }
        remove_file(self.report_path)
        json.dump(thresholds,open(self.report_path,"w"))
        
  






        