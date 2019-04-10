import csv
import torch
from scipy.spatial.distance import euclidean
import random
import shutil
import os
import json
import h5py
import numpy as np
import helpers

from helpers import augment_scene_list


class TorchExtractor():
    def __init__(self,data,torch_params,prepare_params,preprocessing):
    # def __init__(self,data,torch_params):

        data = json.load(open(data))
        torch_params = json.load(open(torch_params))
        prepare_params = json.load(open(prepare_params))
        preprocessing = json.load(open(preprocessing))

        
        # self.prepared_samples = data["prepared_samples_grouped"]
        # self.prepared_labels = data["prepared_labels_grouped"]
        # self.ids_path = data["prepared_ids"]
        # self.samples_ids = json.load(open(self.ids_path))["ids"]
        # self.kept_ids = torch_params["ids_path"]

        self.prepared_images = data["prepared_images"]
        self.stopped_threshold = torch_params["stopped_threshold"]
        self.stopped_prop = torch_params["stopped_prop"]

        self.new_padding = torch_params["new_padding"]
        self.old_padding = torch_params["old_padding"]
        self.smooth = torch_params["smooth"]
        self.smooth_suffix = prepare_params["smooth_suffix"]

        


        prep_toy = prepare_params["toy"]

        if prep_toy:
            self.original_hdf5 = data["hdf5_toy"]
            self.split_hdf5 = torch_params["toy_hdf5"]
            self.max_neighbor_path = torch_params["toy_nb_neighboors_path"]
            self.test_scenes = list(prepare_params["toy_test_scenes"])
            self.train_scenes = list(prepare_params["toy_train_scenes"])
        else:
            self.original_hdf5 = data["hdf5_file"]
            self.split_hdf5 = torch_params["split_hdf5"]
            self.max_neighbor_path = torch_params["nb_neighboors_path"]
            self.test_scenes = list(prepare_params["test_scenes"])
            self.train_scenes = list(prepare_params["train_scenes"])
            self.train_scenes = augment_scene_list(self.train_scenes,preprocessing["augmentation_angles"])
            self.test_scenes = augment_scene_list(self.test_scenes,preprocessing["augmentation_angles"])


        self.seq_len = prepare_params["t_obs"] + prepare_params["t_pred"]
        self.eval_prop = prepare_params["eval_prop"]
        






#

        self.input_size = 2

    def extract_tensors_sophie(self):

        max_neighboors = self.__max_neighbors()

        
        if os.path.exists(self.split_hdf5):
            os.remove(self.split_hdf5)
 
        self.split_dset("test_trajectories",max_neighboors,"trajectories",self.test_scenes,1.0)
        self.split_dset("test_frames",max_neighboors,"frames",self.test_scenes,1.0)

        self.split_dset("train_frames",max_neighboors,"frames",self.train_scenes,self.eval_prop)
        self.split_dset("train_trajectories",max_neighboors,"trajectories",self.train_scenes,self.eval_prop)

        self.split_dset("eval_trajectories",max_neighboors,"trajectories",self.train_scenes,self.eval_prop -1)
        self.split_dset("eval_frames",max_neighboors,"frames",self.train_scenes,self.eval_prop -1)


        if self.smooth:

            self.test_scenes = self.__smooth_scenes(self.test_scenes)
            self.train_scenes = self.__smooth_scenes(self.train_scenes)

            self.split_dset("test_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.test_scenes,1.0)
            self.split_dset("test_frames"+self.smooth_suffix,max_neighboors,"frames",self.test_scenes,1.0)

            self.split_dset("train_frames"+self.smooth_suffix,max_neighboors,"frames",self.train_scenes,self.eval_prop)
            self.split_dset("train_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_scenes,self.eval_prop)

            self.split_dset("eval_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_scenes,self.eval_prop -1)
            self.split_dset("eval_frames"+self.smooth_suffix,max_neighboors,"frames",self.train_scenes,self.eval_prop -1)




        with h5py.File(self.split_hdf5,"r") as dest_file:
            print("in")
            for key in dest_file:
                print(key)
                print(dest_file[key].shape)
        json.dump({"max_neighbors" : max_neighboors},open(self.max_neighbor_path,"w"))


    def __smooth_scenes(self,scenes):
        return [scene + self.smooth_suffix for scene in scenes]        



    def __max_neighbors(self):
        max_n_1 = 0
        # max_n_2 = 0


        with h5py.File(self.original_hdf5,"r") as original_file:
            for key in original_file["trajectories"]:
                dset = original_file["trajectories"][key]
                max_ = dset[0].shape[0]
                if max_ > max_n_1:
                    max_n_1 = max_ 
        # with h5py.File(self.original_hdf5,"r") as original_file:
        #     for key in original_file["frames"]:
        #         dset = original_file["frames"][key]
        #         max_ = dset[0].shape[0]
        #         if max_ > max_n_2:
        #             max_n_2 = max_ 
        # return max_n_1,max_n_2
        return max_n_1


    # if prop > 0 then we take samples up to prop * nb_samples
    # if prop < then we take samples from  prop * nb_samples to the end
    def split_dset(self,name,max_neighboors,sample_type,scene_list,prop):
        with h5py.File(self.original_hdf5,"r") as original_file:
            with h5py.File(self.split_hdf5,"a") as dest_file:
                        # test

                samples = dest_file.create_dataset("samples_{}".format(name),shape=(0,max_neighboors,self.seq_len,2),maxshape = (None,max_neighboors,self.seq_len,2),dtype='float32')
                images = dest_file.create_dataset("images_{}".format(name),shape=(0,),maxshape = (None,),dtype="S20")

                for key in scene_list:
                    
                    dset = original_file[sample_type][key]
                    nb_neighbors = dset[0].shape[0]
                    
                    
                    nb_samples = int(prop*dset.shape[0])
                    scenes = np.array([np.string_(key) for _ in range(np.abs(nb_samples))])
                    print(scenes)

                    padding = np.zeros(shape = (np.abs(nb_samples), max_neighboors-nb_neighbors,self.seq_len,2))

                    
                    samples.resize(samples.shape[0]+np.abs(nb_samples),axis=0)
                    images.resize(images.shape[0]+np.abs(nb_samples),axis=0)

                    if nb_samples > 0:
                        samples[-nb_samples:] = np.concatenate((dset[:nb_samples],padding),axis = 1)
                        images[-nb_samples:] = scenes
                    else:
                        samples[nb_samples:] = np.concatenate((dset[nb_samples:],padding),axis = 1)
                        images[nb_samples:] = scenes


                    
                    


       
    """
    INPUT:
        trajectory: sequence of 2D coordinates
        threshold: distance threshold to be traveled during the trajectory the unit is in normalized scene
        (minmax normalization along each axis)
        returns False if the distance traveled during the trajectory
        is lower than threshold, i.e. the agent has not been moving during the trajectory
    """
    def __is_stopped(self,trajectory ):
        start = trajectory[0]
        end = trajectory[-1]
        d = euclidean(start,end)
        if d < self.stopped_threshold:
            return True
        return False
