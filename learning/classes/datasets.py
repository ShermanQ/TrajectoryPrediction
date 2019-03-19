import torch
from torch.utils import data
import cv2
import numpy as np 
import json
import h5py
import cv2
import time

"""
      set_type:  train eval  test
      use_images: True False
      use_neighbors: True False

      data_type: frames trajectories
"""
class Hdf5Dataset():
      'Characterizes a dataset for PyTorch'
      def __init__(self,data,torch_params,prepare_params,set_type,use_images,data_type,use_neighbors):
            data = json.load(open(data))
            torch_params = json.load(open(torch_params))
            prepare_params = json.load(open(prepare_params))


            self.images_path = data["prepared_images"] + "{}.jpg"
            self.hdf5_file = torch_params["split_hdf5"] 
            self.set_type = set_type

            if set_type == "train" or set_type == "eval":
                  self.scene_list = prepare_params["train_scenes"]
            elif set_type == "test":
                  self.scene_list = prepare_params["test_scenes"]
            else:
                  self.scene_list = []
            self.images = self.__load_images()
            self.data_type = data_type
            self.use_images = use_images
            self.use_neighbors = use_neighbors

            self.dset_name = "samples_{}_{}".format(set_type,data_type)
            self.dset_img_name = "images_{}_{}".format(set_type,data_type)
            self.t_obs = prepare_params["t_obs"]
            self.t_pred = prepare_params["t_pred"]
            self.seq_len = prepare_params["t_obs"] + prepare_params["t_pred"]

      def get_len(self):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  return len(hdf5_file[self.dset_name][:])


      def get_ids(self,ids):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  s = time.time()
                  dset = hdf5_file[self.dset_name]

                  X,y = [],[]
                  if self.use_neighbors:
                        X = dset[ids,:,:self.t_obs]
                        y = dset[ids,:,self.t_obs:self.seq_len]

                  else:
                        X = np.expand_dims( dset[ids,0,:self.t_obs], 1)
                        y = np.expand_dims( dset[ids,0,self.t_obs:self.seq_len], 1)

                  if not self.use_images:
                        print(time.time() - s)
                        return [X,y]

                  dset = hdf5_file[self.dset_img_name]                  
                  imgs = np.array([self.images[img.decode('UTF-8')] for img in dset[ids]]) #224*224*3

                  print(time.time() - s)
                  return [X,y,imgs]

      def __load_images(self):
            images = {}
            for scene in self.scene_list:
                  img = cv2.imread(self.images_path.format(scene))
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
