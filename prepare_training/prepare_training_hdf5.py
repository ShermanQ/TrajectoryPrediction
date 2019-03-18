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

"""
    Consider a trajectory,
    Divide it in sub_trajectories of size t_obs+t_pred, with desired shift
    between sub_trajectories
    Take into account the neighbors that are still in the scene at t_obs
    Do that for every trajectory
"""
class PrepareTrainingHdf5():
    def __init__(self,data,param):
        data = json.load(open(data))
        param = json.load(open(param))
        self.frames_temp = data["temp"] + "frames.txt"
        self.trajectories_temp = data["temp"] + "trajectories.txt"


        self.original_file = data["preprocessed_datasets"] + "{}.csv"
        self.hdf5_dest = data["hdf5_file"]

                 
        with h5py.File(self.hdf5_dest,"a") as f: 
            if "trajectories" not in f:
                f.create_group("trajectories")

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

        max_neighbors = self.__nb_max_neighbors(scene)
        print(max_neighbors)

        helpers.extract_frames(self.original_file.format(scene),self.frames_temp,save = True)
        helpers.extract_trajectories(self.original_file.format(scene),self.trajectories_temp,save = True)

      
        with h5py.File(self.hdf5_dest,"r+") as f:
            # for key in f:
            #     print(key)
            group = f["trajectories"]
            dset = None
            data_shape = (max_neighbors,self.t_obs + self.t_pred,2)

            if scene in group:
                print("in")
                del group[scene] 
            dset = group.create_dataset(scene,shape=(0,data_shape[0],data_shape[1],data_shape[2]),maxshape = (None,data_shape[0],data_shape[1],data_shape[2]),dtype='float32')
            

            with open(self.trajectories_temp) as trajectories:
                with open(self.frames_temp) as file_frames:
                    for k,trajectory in enumerate(trajectories):

                        
                        trajectory = json.loads(trajectory)

                        file_frames,child_iterator = tee(file_frames)
                        
                        frames = trajectory["frames"]
                        current_id = int(trajectory["id"])

                        continuous = self.__are_frames_continuous(frames)

                        if continuous:
                            start,stop = frames[0],frames[-1] + 1             

                            ids = self.__get_neighbors(islice(child_iterator,start,stop))
                            len_traj = len(ids[current_id])

                            for i in range(0,len_traj,self.shift):

                                samples = np.array( self.__samples(len_traj,current_id,ids,i) )

                                nb_neighbors = len(samples)
                                if nb_neighbors != 0:
                                    padding = np.zeros(shape = (max_neighbors-nb_neighbors,data_shape[1],data_shape[2]))
                                    samples = np.concatenate((samples,padding),axis = 0)
                                    dset.resize(dset.shape[0]+1,axis=0)
                                    dset[-1] = samples
                           
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


    def __samples(self,len_traj,current_id,ids,current_frame):
        
        samples = []
        if current_frame + self.t_obs + self.t_pred -1 < len_traj:
            sample = ids[current_id][current_frame:current_frame+self.t_obs+self.t_pred]
            samples.append(sample)

            for id_ in sorted(ids):
                if id_ != current_id:
                    
                    if self.__add_neighbor(ids,id_,current_frame):
                        sample = ids[id_][current_frame:current_frame+self.t_obs+self.t_pred]
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
            True if the neighboor appears during the last time step of the observation time
    """
    def __add_neighbor(self,ids,id_,current_frame):
        if ids[id_][current_frame:current_frame+self.t_obs][-1] != [self.padding,self.padding]:
            return True
        return False

    def __are_frames_continuous(self,frames):
        continuous = True
        nb_frames = (float(frames[-1]-frames[0]))/1.0 + 1.0
        if nb_frames > len(frames):
            continuous = False
        return continuous






# python prepare_training/prepare_training_hdf5.py bookstore0
import numpy as np
def main():
    args = sys.argv

    

    prepare_training = PrepareTrainingHdf5(args[1],args[2])
    
    s = time.time()
    prepare_training.extract_data(args[3])       
    print(time.time()-s)


                    
if __name__ == "__main__":
    main()