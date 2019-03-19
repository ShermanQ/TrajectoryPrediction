import torch
from torch.utils import data
import cv2
import numpy as np 
import json
import h5py
import cv2
import time


class CustomDataLoader():
      def __init__(self,batch_size,shuffle,drop_last,dataset):
            self.shuffle = shuffle 
            self.dataset = dataset
            self.data_len = self.dataset.get_len()
            self.batch_size = batch_size
            self.drop_last = drop_last
            # self.batches = self.__split_batches
            # self.batch_idx = 0
            self.__split_batches()

            # print(self.batches[:3])
      def __split_batches(self):
            self.batches = list(torch.utils.data.BatchSampler(
                  torch.utils.data.RandomSampler(range(self.data_len)),
                  batch_size = self.batch_size,
                  drop_last =self.drop_last))
            self.batch_idx = 0
            self.nb_batches = len(self.batches)
            

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
      def __init__(self,images_path,hdf5_file,scene_list,t_obs,t_pred,set_type,use_images,data_type,use_neighbors):

            self.images_path = images_path + "{}.jpg"
            self.hdf5_file = hdf5_file
            self.set_type = set_type
            self.scene_list = scene_list

            self.images = self.__load_images()
            self.data_type = data_type
            self.use_images = use_images
            self.use_neighbors = use_neighbors

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
                  if self.use_neighbors:
                        X = coord_dset[ids,:,:self.t_obs]
                        y = coord_dset[ids,:,self.t_obs:self.seq_len]
                        

                  else:
                        X = coord_dset[ids,0,:self.t_obs]
                        y = np.expand_dims( coord_dset[ids,0,self.t_obs:self.seq_len], 1)

                        # X = np.expand_dims( coord_dset[ids,0,:self.t_obs], 1)
                        
                        # y = coord_dset[ids,0,self.t_obs:self.seq_len]

                  scenes = [img.decode('UTF-8') for img in scenes_dset[ids]] 

                  if not self.use_images:
                        return (torch.FloatTensor(X),torch.FloatTensor(y),scenes)
                 
                  imgs = torch.stack([self.images[img] for img in scenes_dset[ids]],dim = 0) 
                  
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




"""
      Custom pytorch dataset
      self.list_IDs: ids of sample considered in this dataset
      self.data_path: path whereto read the data

      __getitem__: given an index, selects the corresponding sample id
      and load data and labels files 
      for now return only the trajectory of the main agent, not its neighbors

      It'S done this way so multiprocessing can be used when loading batch with pytorch dataloader
"""
class CustomDatasetSophie(data.Dataset):
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

            with open(self.data_path+"img/img_" + str(ID) + '.txt') as f:
                  img_path = f.read()
                  
                  img = cv2.imread(img_path)
                  
                  
                  img = torch.FloatTensor(img)
                  i,_,c = img.size()
                  img = img.view(c,i,i)
                  X = torch.load(self.data_path + "samples/sample_" + str(ID) + '.pt')
                  y = torch.load(self.data_path + "labels/label_" + str(ID) + '.pt')
            
            return X,img, y, ID

class CustomDatasetIATCNN(data.Dataset):
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

            # with open(self.data_path+"img/img_" + str(ID) + '.txt') as f:
            #       img_path = f.read()
                  
            #       img = cv2.imread(img_path)
                  
                  
            #       img = torch.FloatTensor(img)
            # i,_,c = img.size()
            # img = img.view(c,i,i)
            X = torch.load(self.data_path + "samples/sample_" + str(ID) + '.pt')
            y = torch.load(self.data_path + "labels/label_" + str(ID) + '.pt')

            X = X.permute(2,0,1)

            # xs = X[:,:,0].view(X.size()[0],X.size()[1],1)
            # ys = X[:,:,1].view(X.size()[0],X.size()[1],1)

            # X = torch.cat([xs,ys],dim = 0)
            # X = X.view(X.size()[0],X.size()[1])
            
            # return X,img, y
            return X, y , ID
