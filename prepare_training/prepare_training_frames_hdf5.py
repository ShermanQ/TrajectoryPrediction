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
            f = h5py.File(self.hdf5_dest,"w")
            
        with h5py.File(self.hdf5_dest,"w") as f: 
            f.create_group("frames")

        self.shift = int(param["shift"])
        self.t_obs = int(param["t_obs"])
        self.t_pred = int(param["t_pred"])
        self.padding = param["padding"]
        #





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

        max_neighbors = self.__nb_max_neighbors(scene)
        print(max_neighbors)

        helpers.extract_frames(self.original_file.format(scene),self.frames_temp,save = True)
        
        with h5py.File(self.hdf5_dest,"r+") as f:
            group = f["frames"]
            dset = None
            data_shape = (max_neighbors,self.t_obs + self.t_pred,2)

            if scene in group:
                print("in")
                del group[scene] 
            dset = group.create_dataset(scene,shape=(0,data_shape[0],data_shape[1],data_shape[2]),maxshape = (None,data_shape[0],data_shape[1],data_shape[2]),dtype='float32')
  
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
                            samples = self.__samples(observations[id_])
                            samples = np.array(samples)

                            nb_neighbors = len(samples)
                            if nb_neighbors != 0:
                                padding = np.zeros(shape = (max_neighbors-nb_neighbors,data_shape[1],data_shape[2]))
                                samples = np.concatenate((samples,padding),axis = 0)
                                dset.resize(dset.shape[0]+1,axis=0)
                                dset[-1] = samples
                            
                            
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
    def __samples(self,observations):
        ids = self.__get_neighbors(observations)
        samples = []
        for id_ in sorted(ids):
            if self.__add_neighbor(ids,id_,0):
                sample = ids[id_][0:self.t_obs+self.t_pred]
                samples.append(sample)
        return samples

   


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
        for i,frame in enumerate(frames):
            frame = json.loads(frame)
            frame = frame["ids"]

            for id_ in frame:
                if id_ != "frame":
                    if int(id_) not in ids:
                        ids[int(id_)] = [[self.padding,self.padding] for j in range(i)]
            
            for id_ in ids:
                if str(id_) in frame:
                    ids[id_].append(frame[str(id_)]["coordinates"])
                else:
                    ids[id_].append([self.padding,self.padding])
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
            if p != [self.padding,self.padding]:
                return True
        return add



    def __nb_max_neighbors(self,scene):
        helpers.extract_frames(self.original_file.format(scene),self.frames_temp,save = True)
        nb_agents_scene = []

        with open(self.frames_temp) as frames:
            for i,frame in enumerate(frames):
                # print(frame["ids"])
                frame = json.loads(frame)
                nb_agents = len(frame["ids"].keys())
                nb_agents_scene.append(nb_agents)
        os.remove(self.frames_temp)
        return np.max(nb_agents_scene)


# python prepare_training/prepare_training.py parameters/data.json parameters/prepare_training.json lankershim_inter2


def main():

    args = sys.argv
    prepare_training = PrepareTrainingFramesHdf5(args[1],args[2])
    
    
    s = time.time()
    prepare_training.extract_data(args[3])               
    print(time.time()-s)


                    
if __name__ == "__main__":
    main()