import torch
from torch.utils import data
import cv2
import numpy as np 
import json
import h5py
import cv2
import time


class CustomDataLoader():
      def __init__(self,batch_size,shuffle,drop_last,dataset,test = 0):
            self.shuffle = shuffle 
            self.dataset = dataset
            self.data_len = self.dataset.get_len()
            self.batch_size = batch_size
            self.drop_last = drop_last
            # self.batches = self.__split_batches
            # self.batch_idx = 0
            self.test = test
            self.__split_batches()
            

            # print(self.batches[:3])
      def __split_batches(self):
            torch.manual_seed(100)
            self.batches = list(torch.utils.data.BatchSampler(
                  torch.utils.data.RandomSampler(range(self.data_len)),
                  batch_size = self.batch_size,
                  drop_last =self.drop_last))
            self.batch_idx = 0
            self.nb_batches = len(self.batches)
            print(self.nb_batches)
            if self.test :
                  self.nb_batches = 10

            

      def __iter__(self):
            return self
      def __next__(self):

            if self.batch_idx >= self.nb_batches:
                  self.__split_batches()
                  raise StopIteration
            else:     
                  ids = sorted(self.batches[self.batch_idx])
                  self.batch_idx += 1 
                  return self.dataset.get_ids(ids)

"""
      set_type:  train eval  test
      use_images: True False
      use_neighbors: True False

      data_type: frames trajectories
"""
class Hdf5Dataset():
      'Characterizes a dataset for PyTorch'
      def __init__(self,images_path,hdf5_file,scene_list,t_obs,t_pred,set_type,use_images,data_type,use_neighbors_sample,use_neighbors_label,reduce_batches = True):

            self.images_path = images_path + "{}.jpg"
            self.hdf5_file = hdf5_file
            self.set_type = set_type
            self.scene_list = scene_list

            self.images = self.__load_images()
            self.data_type = data_type
            self.use_images = use_images
            self.use_neighbors_sample = use_neighbors_sample
            self.use_neighbors_label = use_neighbors_label
            self.reduce_batches = reduce_batches


            self.dset_name = "samples_{}_{}".format(set_type,data_type)
            self.dset_img_name = "images_{}_{}".format(set_type,data_type)
            self.t_obs = t_obs
            self.t_pred = t_pred
            self.seq_len = t_obs + t_pred

      def get_len(self):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  return len(hdf5_file[self.dset_name][:])


      def get_ids(self,ids):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  coord_dset = hdf5_file[self.dset_name]
                  scenes_dset = hdf5_file[self.dset_img_name]   
             


                  X,y,scenes = [],[],[]
                  max_batch = 0
                  if self.use_neighbors_sample:
                        X = coord_dset[ids,:,:self.t_obs]


                        if self.reduce_batches:
                              b,n,s,i = X.shape
                              nb_agents = np.sum( np.sum(X.reshape(b,n,-1),axis = 2) > 0, axis = 1 )
                              
                              max_batch = np.max(nb_agents)
                              
                              X = X[:,:max_batch,:,:]


                  else: 
                        X = coord_dset[ids,0,:self.t_obs] 
                        
                  if self.use_neighbors_label:     
                        y = coord_dset[ids,:,self.t_obs:self.seq_len]
                        if self.reduce_batches:
                              y = y[:,:max_batch,:,:]
                              
                  else:                        
                        y = np.expand_dims( coord_dset[ids,0,self.t_obs:self.seq_len], 1)


                  scenes = [img.decode('UTF-8') for img in scenes_dset[ids]] 


                  if not self.use_images:
                        return (torch.FloatTensor(X),torch.FloatTensor(y),scenes)
                 
                  imgs = torch.stack([self.images[img] for img in scenes],dim = 0) 
                  
                  return (torch.FloatTensor(X),torch.FloatTensor(y),imgs,scenes)

      def __load_images(self):
            images = {}
            for scene in self.scene_list:
                  img = torch.FloatTensor(cv2.imread(self.images_path.format(scene)))
                  img = img.permute(2,0,1)
                  images[scene] = img
            return images


        


"""
      Custom pytorch dataset
      self.list_IDs: ids of sample considered in this dataset
      self.data_path: path whereto read the data

      __getitem__: given an index, selects the corresponding sample id
      and load data and labels files 
      for now return only the trajectory of the main agent, not its neighbors

      It'S done this way so multiprocessing can be used when loading batch with pytorch dataloader
"""
import time
class CustomDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs,data_path):
        'Initialization'
        self.list_IDs = np.array(list_IDs)
        self.data_path = data_path

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
      #   X = torch.load(self.data_path + "samples/sample" + str(ID) + '.pt')[0].view(-1)
      #   y = torch.load(self.data_path + "labels/label" + str(ID) + '.pt')[0].view(-1)

      #   X = torch.load(self.data_path + "samples/sample" + str(ID) + '.pt')[0]
      #   y = torch.load(self.data_path + "labels/label" + str(ID) + '.pt')[0]
        X = torch.load(self.data_path + "samples/sample_" + str(ID) + '.pt')[0]
      #   print(X.size())
        y = torch.load(self.data_path + "labels/label_" + str(ID) + '.pt')[0]
        y = y.unsqueeze(0)
        
        
      #   print("x")
      #   print(X)
      #   print("y")
      #   print(y)
        return X, y, ID
