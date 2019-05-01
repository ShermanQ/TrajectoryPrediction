from helpers import extract_trajectories,get_speeds,get_accelerations,remove_file
import json
import numpy as np
import matplotlib.pyplot as plt

class Accelerations():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))

        self.scenes = self.prepare_params["selected_scenes"]
        self.scene_paths = self.data_params["filtered_datasets"] + "{}.csv"
        self.report_path = self.data_params["dynamics"]
        self.trajectories_temp = self.data_params["temp"] + "trajectories.txt"
        self.delta_t = 1.0/float(self.prepare_params["framerate"])


    def get_distribs(self):
        types_dict = {}
        dataset_accelerations = []
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
                    trajectory_len.append(len(coordinates))
                    accelerations = get_accelerations(get_speeds(coordinates,self.delta_t),self.delta_t)

                    scene_accelerations.append(accelerations)
            scene_accelerations = np.concatenate(scene_accelerations)

            # plt.hist(scene_accelerations,bins = 100)
            # plt.show()

            dataset_accelerations.append(scene_accelerations)
        
        dataset_accelerations = np.concatenate(dataset_accelerations)
        percentile = np.percentile(dataset_accelerations,[1,99])
        print(percentile)
        print(np.mean(trajectory_len),ctr)
        plt.hist(dataset_accelerations,bins = 100,label= "acceleration distribution")
        plt.legend()
        plt.show()

        return dataset_accelerations





        