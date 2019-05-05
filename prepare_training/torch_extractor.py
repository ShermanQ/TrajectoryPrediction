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
from sklearn.preprocessing import OneHotEncoder

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
        self.padding = prepare_params["padding"]

        self.smooth = torch_params["smooth"]
        self.smooth_suffix = prepare_params["smooth_suffix"]

        self.pedestrian_only = prepare_params["pedestrian_only"]
        self.original_dataset = prepare_params["original"]

        
        


        prep_toy = prepare_params["toy"]

        if prep_toy:
            self.original_hdf5 = data["hdf5_toy"]
            self.split_hdf5 = torch_params["toy_hdf5"]
            self.max_neighbor_path = torch_params["toy_nb_neighboors_path"]
            self.test_scenes = list(prepare_params["toy_test_scenes"])
            self.train_eval_scenes = list(prepare_params["toy_train_scenes"])
            self.eval_scenes = list(prepare_params["toy_test_scenes"])
            self.train_scenes = [ scene for scene in self.train_eval_scenes if scene not in self.eval_scenes]
        else:

            self.original_hdf5 = data["hdf5_file"]
            self.split_hdf5 = torch_params["split_hdf5"]
            self.max_neighbor_path = torch_params["nb_neighboors_path"]

            if self.pedestrian_only:
                self.original_hdf5 = data["hdf5_ped"]
            #     self.split_hdf5 = torch_params["ped_hdf5"]
            #     self.max_neighbor_path = torch_params["nb_pedestrians_path"]

            
                

            self.test_scenes = list(prepare_params["test_scenes"])
            self.train_eval_scenes = list(prepare_params["train_scenes"])
            self.eval_scenes = list(prepare_params["eval_scenes"])
            self.train_scenes = [ scene for scene in self.train_eval_scenes if scene not in self.eval_scenes]

            if self.original_dataset:
                self.test_scenes = list(prepare_params["test_original"])
                self.train_eval_scenes = list(prepare_params["train_original"])
                self.eval_scenes = list(prepare_params["eval_original"])
                self.train_scenes = [ scene for scene in self.train_eval_scenes if scene not in self.eval_scenes]

            # self.train_scenes = augment_scene_list(self.train_scenes,preprocessing["augmentation_angles"])
            # self.test_scenes = augment_scene_list(self.test_scenes,preprocessing["augmentation_angles"])


        self.seq_len = prepare_params["t_obs"] + prepare_params["t_pred"]
        self.eval_prop = prepare_params["eval_prop"]
        self.nb_types = len(prepare_params["types_dic"].keys()) + 1
        print(self.nb_types)
        cat = np.arange(self.nb_types).reshape(self.nb_types,1)
        print(cat)
        self.ohe = OneHotEncoder(sparse = False,categories = "auto")
        self.ohe = self.ohe.fit(cat)

        print(self.ohe.categories_)





#

        self.input_size = 2

    def extract_tensors_sophie(self):

        max_neighboors = self.__max_neighbors()

        
        if os.path.exists(self.split_hdf5):
            os.remove(self.split_hdf5)
 
        self.split_dset("test_trajectories",max_neighboors,"trajectories",self.test_scenes,1.0)
        self.split_dset("train_eval_trajectories",max_neighboors,"trajectories",self.train_eval_scenes,1.0)
        self.split_dset("train_trajectories",max_neighboors,"trajectories",self.train_scenes,1.0)
        self.split_dset("eval_trajectories",max_neighboors,"trajectories",self.eval_scenes,1.0)





        # self.split_dset("test_frames",max_neighboors,"frames",self.test_scenes,1.0)
        # self.split_dset("eval_frames",max_neighboors,"frames",self.train_scenes,self.eval_prop -1)
        # self.split_dset("train_frames",max_neighboors,"frames",self.train_scenes,self.eval_prop)



        if self.smooth:

            self.test_scenes = self.__smooth_scenes(self.test_scenes)
            self.train_scenes = self.__smooth_scenes(self.train_scenes)
            self.train_eval_scenes = self.__smooth_scenes(self.train_eval_scenes)
            self.eval_scenes = self.__smooth_scenes(self.eval_scenes)


            self.split_dset("test_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.test_scenes,1.0)
            self.split_dset("train_eval_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_eval_scenes,1.0)
            self.split_dset("train_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_scenes,1.0)
            self.split_dset("eval_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.eval_scenes,1.0)

            

            # self.split_dset("test_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.test_scenes,1.0)
            # self.split_dset("train_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_scenes,self.eval_prop)
            # self.split_dset("eval_trajectories"+self.smooth_suffix,max_neighboors,"trajectories",self.train_scenes,self.eval_prop -1)

            # self.split_dset("test_frames"+self.smooth_suffix,max_neighboors,"frames",self.test_scenes,1.0)
            # self.split_dset("train_frames"+self.smooth_suffix,max_neighboors,"frames",self.train_scenes,self.eval_prop)
            # self.split_dset("eval_frames"+self.smooth_suffix,max_neighboors,"frames",self.train_scenes,self.eval_prop -1)




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




                samples = dest_file.create_dataset("samples_{}".format(name),shape=(0,max_neighboors,self.seq_len,2),maxshape = (None,max_neighboors,self.seq_len,2),dtype='float32',chunks=(15,max_neighboors,self.seq_len,2))
                images = dest_file.create_dataset("images_{}".format(name),shape=(0,),maxshape = (None,),dtype="S20")
                types = dest_file.create_dataset("types_{}".format(name),shape=(0,max_neighboors,self.nb_types-1),maxshape = (None,max_neighboors,self.nb_types-1),dtype='float32')

                for key in scene_list:
                    
                    dset = original_file[sample_type][key]
                    dset_types = original_file[sample_type][key+"_types"]

                    nb_neighbors = dset[0].shape[0]
                    
                    
                    nb_samples = int(prop*dset.shape[0])
                    scenes = np.array([np.string_(key) for _ in range(np.abs(nb_samples))])
                    # print(scenes)

                    padding = np.ones(shape = (np.abs(nb_samples), max_neighboors-nb_neighbors,self.seq_len,2))
                    padding = padding * self.padding
                    padding_types = np.zeros(shape = (np.abs(nb_samples), max_neighboors-nb_neighbors,self.nb_types - 1))


                    
                    samples.resize(samples.shape[0]+np.abs(nb_samples),axis=0)
                    images.resize(images.shape[0]+np.abs(nb_samples),axis=0)
                    types.resize(types.shape[0]+np.abs(nb_samples),axis=0)


                    

                    if nb_samples > 0:
                        samples[-nb_samples:] = np.concatenate((dset[:nb_samples],padding),axis = 1)

                        ohe_types = np.array([self.ohe.transform(d.reshape(-1,1))  for d in dset_types[:nb_samples]])
                        types[-nb_samples:] = np.concatenate((ohe_types[:,:,1:],padding_types),axis = 1)

                        images[-nb_samples:] = scenes
                    else:
                        samples[nb_samples:] = np.concatenate((dset[nb_samples:],padding),axis = 1)


                        ohe_types = np.array([self.ohe.transform(d.reshape(-1,1))  for d in dset_types[nb_samples:]])
                        types[nb_samples:] = np.concatenate((ohe_types[:,:,1:],padding_types),axis = 1)

                        images[nb_samples:] = scenes


                    
                    #dset_types[:nb_samples] ohe


       
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
