import csv
from itertools import islice
import helpers 
import json
import numpy as np
import time
from itertools import tee
import os
import sys

"""
    Consider a trajectory,
    Divide it in sub_trajectories of size t_obs+t_pred, with desired shift
    between sub_trajectories
    Take into account the neighbors that are still in the scene at t_obs
    Do that for every trajectory
"""
class PrepareTraining():
    def __init__(self,data,param):
        data = json.load(open(data))
        param = json.load(open(param))
        self.frames_temp = data["temp"] + "frames.txt"
        self.trajectories_temp = data["temp"] + "trajectories.txt"
        self.original_file = data["preprocessed_datasets"] + "{}.csv"
        self.labels_dest = data["prepared_labels"] + "{}.csv"
        self.samples_dest = data["prepared_samples"] + "{}.csv"
        self.shift = int(param["shift"])
        self.t_obs = int(param["t_obs"])
        self.t_pred = int(param["t_pred"])




    """
        parameters: dict containing reauired parameters[
            "original_file" path for file containing the original data
            "frames_temp"   path to store temporarily the frame-shaped data extracted from original_file
            "trajectories_temp" path to store temporarily the trajectory-shaped data extracted from original_file
            "shift" the size of the step between two feature extraction for the main trajectory
            t_obs: number of observed frames
            t_pred: number of frames to predict
            "scene" scene name
            "framerate" framerate of the original data
            "data_path path for file where to write down features
            "label_path" path for file where to write down labels
        ]
    """
    def extract_data(self,scene):

        helpers.extract_frames(self.original_file.format(scene),self.frames_temp,save = True)
        helpers.extract_trajectories(self.original_file.format(scene),self.trajectories_temp,save = True)

        sample_id = 0

        helpers.remove_file(self.samples_dest.format(scene))
        helpers.remove_file(self.labels_dest.format(scene))


        with open(self.samples_dest.format(scene),"a") as data_csv:
            data_writer = csv.writer(data_csv)
            with open(self.labels_dest.format(scene),"a") as label_csv:
                label_writer = csv.writer(label_csv)
                # with open(parameters["scene_path"],"a") as scene_csv:
                #     scene_writer = csv.writer(scene_csv)
                with open(self.trajectories_temp) as trajectories:
                    with open(self.frames_temp) as file_frames:
                        for k,trajectory in enumerate(trajectories):

                            
                            trajectory = json.loads(trajectory)

                            scene_name = trajectory["scene"]
                            file_frames,a = tee(file_frames)
                            
                            frames = trajectory["frames"]
                            current_id = int(trajectory["id"])
                            # if current_id == 63:
                            #     print("63")
                            continuous = self.__are_frames_continuous(frames)

                            if continuous:
                                start,stop = frames[0],frames[-1] + 1

                                

                                ids = self.__get_neighbors(islice(a,start,stop))
                                len_traj = len(ids[current_id])

                                for i in range(0,len_traj,self.shift):
                                    
                                    features,labels = self.__features_labels(len_traj,current_id,ids,i)
                                    if features != []:
                                        sample_id = self.__persist_data(ids,current_id,i,features,labels,data_writer,label_writer,sample_id,scene_name)
                                # if sample_id == 39:
                                #     print("")
                            else:
                                print("trajectory {} discarded".format(current_id))
        
        os.remove(self.frames_temp)
        os.remove(self.trajectories_temp)

    """
    in:
        len_traj: length of the main trajectory
        shift: the size of the step between to feature extraction for the main trajectory
        t_obs: number of observed frames
        t_pred: number of frames to predict
        current_id: id of the main trajectory
        ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        current_frame: the frame of the trajectory to be considered
    out:
        features: first to appear is the main trajectory, then the neighboors that appears during observation
                  everything is flattened: let xij, i traj_id, j time_step [x00,y00,...x0n,y0n, ... , xn0,yn0,...xnn,ynn,]
                  j < current_frame + t_obs
        labels: same idea but for prediction time


    """
    def __features_labels(self,len_traj,current_id,ids,current_frame):
        
        features = []
        labels = []
        if current_frame + self.t_obs + self.t_pred -1 < len_traj:
            

            feature,label = self.__feature_label(ids,current_id,current_frame)

            # if feature[0] == -1:
            #     print("fdggsdg")

            features.append(feature)
            labels.append(label)

            for id_ in sorted(ids):
                if id_ != current_id:
                    
                    if self.__add_neighbor(ids,id_,current_frame):


                        feature,label = self.__feature_label(ids,id_,current_frame)

                        features.append(feature)
                        labels.append(label)

            features = np.array(features).flatten().tolist()
            labels = np.array(labels).flatten().tolist()
        return features,labels

    """
    in:
        ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        id_: id of the neighbor to test
        
        t_obs: number of observed frames
        t_pred: number of frames to predict
        current_frame: the frame of the trajectory to be considered
    out: 
        feature: for the given id, steps between current_frame and current_frame + t_obs 2D COORDINATES ARE FLATTENED
        label: for the given id, steps between current_frame+t_obs and current_frame+t_obs+t_pred

    """
    def __feature_label(self,ids,id_,current_frame,padding = -1):

        feature = ids[id_][current_frame:current_frame+self.t_obs]
        ctr = 0
        for p in feature:
            if p[0] == padding:
                ctr += 1
        feature = np.array(feature[ctr:] + feature[:ctr])
        feature = feature.flatten().tolist()
        label = np.array(ids[id_][current_frame+self.t_obs:current_frame+self.t_obs+self.t_pred]).flatten().tolist()
        return feature,label
    """
        in: 
            start:frame number where the trajectory begins
            stop: frame number where the trajectory stops
            frames_path: path to the file containing frames
        out:
            returns ids but the lists are filled with the coordinates
            of theirs objects for a given frame or [-1,-1] if its not in 
            the given frame
    """
    def __get_neighbors(self,frames):
        ids = {}
        # with open(frames_path) as frames:
        for i,frame in enumerate(frames):
            frame = json.loads(frame)
            frame = frame["ids"]

            for id_ in frame:
                # if id_ != "frame":
                if int(id_) not in ids:
                    ids[int(id_)] = [[-1,-1] for j in range(i)]
            
            for id_ in ids:
                if str(id_) in frame:
                    ids[id_].append(frame[str(id_)]["coordinates"])
                else:
                    ids[id_].append([-1,-1])
        return ids


    """
        in:
            t_obs: number of observed frames
            ids: ids  filled with the coordinates
            id_: id of the neighbor to test
            of theirs objects for a given frame or [-1,-1] if its not in 
            the given frame
            current_frame: the frame of the trajectory to be considered
        out: 
            True if the neighboor appears during the last time step of the observation time
    """
    def __add_neighbor(self,ids,id_,current_frame):
        if ids[id_][current_frame:current_frame+self.t_obs][-1] != [-1,-1]:
            return True
        return False

    def __are_frames_continuous(self,frames):
        continuous = True
        nb_frames = (float(frames[-1]-frames[0]))/1.0 + 1.0
        if nb_frames > len(frames):
            continuous = False
        return continuous



    """
        in:
            ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
            the given frame
            features: see features_labels
            labels: see features_labels
            data_writer: csv writer for file containing features
            label_writer: csv writer for file containing labels
            sample_id: running id of the number of training sample
        out:
    """
    def __persist_data(self,ids,current_id,current_frame,features,labels,data_writer,label_writer,sample_id,scene):
        

        # ids_list = sorted([id_ for id_ in ids if add_neighbor(parameters["t_obs"],ids,id_,current_frame)])
        # nb_objects = len(ids_list)
        nb_objects = int((len(features)/2)/self.t_obs)
        features_header = [
            sample_id,
            nb_objects,
            # ids_list,
            self.t_obs,
            self.t_pred
            # start,
            # stop,
            # parameters["scene"],
            # parameters["framerate"]

        ]

        labels_header = [
            sample_id
        ]
        features = features_header + features
        labels = labels_header + labels

        sample_id += 1

        data_writer.writerow(features)
        label_writer.writerow(labels)
    #
        return sample_id




# python prepare_training/prepare_training.py parameters/data.json parameters/prepare_training.json lankershim_inter2

def main():
    args = sys.argv

    prepare_training = PrepareTraining(args[1],args[2])
    
    s = time.time()
    prepare_training.extract_data(args[3])       
    print(time.time()-s)


                    
if __name__ == "__main__":
    main()