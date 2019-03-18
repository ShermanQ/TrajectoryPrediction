import csv
from itertools import islice
import helpers 
import json
import numpy as np
import time
from itertools import tee
import os
import sys
import h5py


class PrepareTrainingFramesHdf5():
    def __init__(self,data,param):
        data = json.load(open(data))
        param = json.load(open(param))
        self.frames_temp = data["temp"] + "frames.txt"
        self.original_file = data["preprocessed_datasets"] + "{}.csv"

        self.hdf5_dest = data["hdf5_file"]

        if not os.path.isfile(self.hdf5_dest):
            print("in")
            with h5py.File(self.hdf5_dest,"w") as f: 
                f.create_group("frames")

        self.shift = int(param["shift"])
        self.t_obs = int(param["t_obs"])
        self.t_pred = int(param["t_pred"])
        self.padding = param["padding"]





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
        
        helpers.remove_file(self.samples_dest.format(scene))
        helpers.remove_file(self.labels_dest.format(scene))

        

        with open(self.samples_dest.format(scene),"a") as data_csv:
            data_writer = csv.writer(data_csv)
            with open(self.labels_dest.format(scene),"a") as label_csv:
                label_writer = csv.writer(label_csv)

                with open(self.frames_temp) as frames:
                    observations = {}
                    sample_id = 0
                    for frame in frames:
                        delete_ids = []
                        observations[sample_id] = []
                        sample_id += 1

                        for id_ in observations:
                            if len(observations[id_]) < self.t_obs + self.t_pred:
                                observations[id_].append(frame)
                            else:
                                features,labels,ids = self.__features_labels(observations[id_])
                                self.__persist_data(ids,features,labels,data_writer,label_writer,id_)
                                
                                delete_ids.append(id_)
                        for id_ in delete_ids:
                            del observations[id_]
        helpers.remove_file(self.frames_temp)
    """
    in:
        observations: list of frames size t_obs+t_pred
        t_obs: number of observed frames
        t_pred: number of frames to predict
        
    out:
        features: for the list of t_obs+t_pred frames, for each id in the sequence
        if its coordinates are not all [-1,-1] during observation time, its coordinates
        during observation time are flattened and added
        
        labels: same idea but for prediction time
    """
    def __features_labels(self,observations):
        ids = self.__get_neighbors(observations)
        features = []
        labels = []
        for id_ in sorted(ids):
            if self.__add_neighbor(ids,id_,0):
                feature,label = self.__feature_label(ids,id_,0)
                features.append(feature)
                labels.append(label)
        features = np.array(features).flatten().tolist()
        labels = np.array(labels).flatten().tolist()
        return features,labels,ids

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
    def __feature_label(self,ids,id_,current_frame):
        feature = np.array(ids[id_][current_frame:current_frame+self.t_obs]).flatten().tolist()
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
    # def get_neighbors_coordinates(start,stop,frames,ids):
        ids = {}
        # with open(frames_path) as frames:
        for i,frame in enumerate(frames):
            frame = json.loads(frame)
            frame = frame["ids"]

            for id_ in frame:
                if id_ != "frame":
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
            True if the neighboor appears during observation time
    """
    def __add_neighbor(self,ids,id_,current_frame):
        add = False
        for p in ids[id_][current_frame:current_frame+self.t_obs]:
            if p != [-1,-1]:
                return True
        return add



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
    def __persist_data(self,ids,features,labels,data_writer,label_writer,sample_id):
        nb_objects = int((len(features)/2)/self.t_obs)
        # ids_list = sorted([id_ for id_ in ids])
        # nb_objects = len(ids_list)
        features_header = [
            sample_id,
            nb_objects,
            # ids_list,
            self.t_obs,
            self.t_pred,
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

        return sample_id


# python prepare_training/prepare_training.py parameters/data.json parameters/prepare_training.json lankershim_inter2


def main():

    args = sys.argv
    prepare_training = PrepareTrainingFrames(args[1],args[2])
    
    
    s = time.time()
    prepare_training.extract_data(args[3])               
    print(time.time()-s)


                    
if __name__ == "__main__":
    main()