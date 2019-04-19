import torch
from torch.utils import data
import cv2
import numpy as np 
import json
import h5py
import cv2
import time
import helpers


class CustomDataLoader():
      def __init__(self,batch_size,shuffle,drop_last,dataset,test = 0):
            self.shuffle = shuffle 
            self.dataset = dataset
            # print("len")
            self.data_len = self.dataset.get_len()
            # print(self.data_len)

            self.batch_size = batch_size
            self.drop_last = drop_last
            # self.batches = self.__split_batches
            # self.batch_idx = 0
            self.test = test
            self.split_batches()
            

            # print(self.batches[:3])
      def split_batches(self):
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
                  self.split_batches()
                  raise StopIteration
            else:     
                  ids = sorted(self.batches[self.batch_idx])
                  self.batch_idx += 1 
                  return self.dataset.get_ids(ids)



"""
      set_type:  train eval  test
      use_images: True False
      use_neighbors: True False
      predict_offsets: 0: none, 1: based on last obs point, 2: based on previous point

      data_type: frames trajectories
"""
class Hdf5Dataset():
      'Characterizes a dataset for PyTorch'
      def __init__(self,padding,images_path,hdf5_file,scene_list,t_obs,t_pred,set_type,use_images,data_type,use_neighbors,augmentation,augmentation_angles,centers,reduce_batches = True,predict_offsets = 0,predict_smooth=0,smooth_suffix = ""):

            self.images_path = images_path + "{}.jpg"
            self.hdf5_file = hdf5_file
            self.set_type = set_type
            self.scene_list = scene_list

            self.data_type = data_type
            self.use_images = use_images
            self.use_neighbors = use_neighbors

            
            self.centers = centers
            self.reduce_batches = reduce_batches
            self.predict_offsets = predict_offsets

            self.predict_smooth = predict_smooth
            self.smooth_suffix = smooth_suffix


            self.dset_name = "samples_{}_{}".format(set_type,data_type)
            self.dset_img_name = "images_{}_{}".format(set_type,data_type)
            self.dset_types = "types_{}_{}".format(set_type,data_type)

            self.t_obs = t_obs
            self.t_pred = t_pred
            self.seq_len = t_obs + t_pred
            self.augmentation = augmentation 
            self.augmentation_angles = augmentation_angles
            self.padding = padding
            

            if self.augmentation:
                  self.r_matrices = self.__get_matrices()
                  self.scene_list = helpers.helpers_training.augment_scene_list(self.scene_list,self.augmentation_angles)
            
            self.images = self.__load_images()

      def get_len(self):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  if self.augmentation:
                        return hdf5_file[self.dset_name].shape[0]  * (len(self.augmentation_angles) + 1)
                  return hdf5_file[self.dset_name].shape[0]

      


      def get_ids(self,ids):
            with h5py.File(self.hdf5_file,"r") as hdf5_file: 
                  coord_dset = hdf5_file[self.dset_name]
                  scenes_dset = hdf5_file[self.dset_img_name]   
                  types_dset = hdf5_file[self.dset_types]   


                  X,y,types,m_ids = [],[],[],[]
                  max_batch = coord_dset.shape[1]



                  if self.augmentation:
                        ids,m_ids = self.__augmentation_ids(ids,coord_dset)
                  
                  scenes = [img.decode('UTF-8') for img in scenes_dset[ids]] # B
                  
                        

                  if self.reduce_batches:
                        max_batch = self.__get_batch_max_neighbors(ids,coord_dset)


                  if self.use_neighbors:
                        X,y,types = self.__get_x_y_neighbors(coord_dset,ids,max_batch,types_dset,hdf5_file)

                  else:
                        X,y,types = self.__get_x_y(coord_dset,ids,max_batch,types_dset,hdf5_file)                      



                  if self.augmentation:
                        X,y = self.__augment_batch(scenes,X,y,m_ids)
                        scenes = [scene if m == 0 else scene +"_{}".format(m) for scene,m in zip(scenes,m_ids)] # B

                        
                  if not self.use_images:
                        return (torch.FloatTensor(X).contiguous(),torch.FloatTensor(y).contiguous(),scenes,torch.FloatTensor(types))
            
                  imgs = torch.stack([self.images[img] for img in scenes],dim = 0) 
                  
                  return (torch.FloatTensor(X).contiguous(),torch.FloatTensor(y).contiguous(),imgs,scenes,torch.FloatTensor(types))


      def __get_batch_max_neighbors(self,ids,coord_dset):
            b,n,s,i = coord_dset.shape

            active_mask = (coord_dset[ids,:,:self.t_obs] == self.padding).astype(int)
            a = np.sum(active_mask,axis = 3)
            b = np.sum( a, axis = 2)
            nb_padding_traj = b/float(2.0*self.t_obs) #prop of padded points per traj
            active_traj = nb_padding_traj < 1.0 # if less than 100% of the points are padding points then its an active trajectory
            nb_agents = np.sum(active_traj.astype(int),axis = 1)                      
            max_batch = np.max(nb_agents)
            return max_batch


      def __get_x_y_neighbors(self,coord_dset,ids,max_batch,types_dset,hdf5_file):
            X = coord_dset[ids,:max_batch,:self.t_obs] # load obs for given ids

            if self.predict_smooth:
                  coord_dset = hdf5_file[self.dset_name+self.smooth_suffix] # if using smoothed trajectories for prediction, switch dataset
            
            y = coord_dset[ids,:max_batch,self.t_obs:self.seq_len] #B,N,tpred,2 load pred for given ids
            types = types_dset[ids,:max_batch] #B,N,tpred,2

            
            if self.predict_offsets:


                  if self.predict_offsets == 1:
                        # offsets according to last obs point, take last point for each obs traj and make it an array of dimension y
                        last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2)#B,N,tpred,2
                  elif self.predict_offsets == 2:# y shifted left

                        # offsets according to preceding point point, take points for tpred shifted 1 timestep left
                        last_points = coord_dset[ids,:max_batch,self.t_obs-1:self.seq_len-1]

                  active_mask = (y != self.padding).astype(int)
                  ####################################################


                  # mauvaise selection car un coordonnée peut être égal à 0
                  # et sera considéré comme du padding
                  # mettre le padding à -1 pas de conflit pour les valeurs x,y absolues
                  # après offset -1 est une valeur possible prises par les offsets
                  # 0 aussi,
                  # question padding -1 avant dataset --> permet identification facile
                  # après dataset









                  #####################################################
                  active_last_points = np.multiply(active_mask,last_points)
                  y = np.subtract(y,active_last_points)
            return X,y,types

      def __get_x_y(self,coord_dset,ids,max_batch,types_dset,hdf5_file):
            X = np.expand_dims( coord_dset[ids,0,:self.t_obs] ,1) # keep only first neighbors and expand nb_agent dim 

            if self.predict_smooth: # if predict smoothed target change dataset
                  coord_dset = hdf5_file[self.dset_name+self.smooth_suffix]   
                  
            y = np.expand_dims( coord_dset[ids,0,self.t_obs:self.seq_len], 1) #B,1,tpred,2 # keep only first neighbors and expand nb_agent dim 
            types =  types_dset[ids,0] #B,1,tpred,2

            if self.predict_offsets:



                  if self.predict_offsets == 1 :
                        # last_points = np.repeat(  np.expand_dims(np.expand_dims(X,1)[:,:,-1],2),  self.t_pred, axis=2) #B,1,tpred,2
                        last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2) #B,1,tpred,2
                  
                  elif self.predict_offsets == 2: # y shifted left
                        last_points = np.expand_dims( coord_dset[ids,0,self.t_obs-1:self.seq_len-1], 1)

                  active_mask = (y != self.padding).astype(int)
                  active_last_points = np.multiply(active_mask,last_points)
                  y = np.subtract(y,active_last_points)
            return X,y,types

      def __augmentation_ids(self,ids,coord_dset):
            red_ids = sorted(np.array(ids) % coord_dset.shape[0])
            m_ids = (np.array(ids) / float(coord_dset.shape[0]) ).astype(int)*90
            ids,matrix_indexes = [],[]

            for i in range(len(red_ids)):                              
                  if i > 0 and  red_ids[i] == ids[-1]:
                        ids.append(red_ids[i]+1)
                  else:
                        ids.append(red_ids[i])
            return ids,m_ids

      def __augment_batch(self,scenes,X,y,m_ids):
            centers = np.array([self.centers[scene] for scene in scenes]) # B,2
            centers = np.expand_dims(centers,axis = 1) # B,1,2
            centers = np.expand_dims(centers,axis = 1) # B,1,1,2
            centers = np.repeat(centers,X.shape[1],axis = 1) # B,N,1,2

            centers_x = np.repeat(centers,X.shape[2],axis = 2) # B,N,t_obs,2
            centers_y = np.repeat(centers,y.shape[2],axis = 2) # B,N,t_pred,2

            centers_x = np.multiply( (X != self.padding).astype(int), centers_x) # put 0 centers where padding points
            centers_y = np.multiply( (y != self.padding).astype(int), centers_y)

            matrices = np.array([self.r_matrices[m] for m in m_ids]) #B,2,2
            matrices = np.expand_dims(matrices,axis = 1) #B,1,2,2
            matrices = np.repeat(matrices,X.shape[1],axis = 1) #B,N,2,2

            matrices_x = np.repeat( np.expand_dims(matrices,axis = 2), X.shape[2],axis=2) #B,N,tobs,2,2
            matrices_y = np.repeat( np.expand_dims(matrices,axis = 2), y.shape[2],axis=2) #B,N,pred,2,2

            matrices_x = np.multiply( np.expand_dims((X != self.padding).astype(int),4), matrices_x) # put 0 matrices where padding points on x
            matrices_y = np.multiply( np.expand_dims( (y != self.padding).astype(int),4), matrices_y) # put 0 matrices where padding points on y


            eyes = np.expand_dims(np.expand_dims(np.expand_dims(np.eye(X.shape[-1]), 0),0),0) # create identity matrix of dimension 1,1,1,2,2
            eyes_x = eyes.repeat(X.shape[0],0).repeat(X.shape[1],1).repeat(X.shape[2],2) # B,N,tobs,2,2
            eyes_y = eyes.repeat(y.shape[0],0).repeat(y.shape[1],1).repeat(y.shape[2],2) # B,N,tpred,2,2

            eyes_x = np.multiply( np.expand_dims( (X == self.padding).astype(int) ,4), eyes_x) # put 0 matrices where normal points on x
            eyes_y = np.multiply( np.expand_dims( (y == self.padding).astype(int) ,4), eyes_y) # put 0 matrices where normal points on y

            matrices_x = np.add(matrices_x,eyes_x) # identity on padding points, rotation on normal points
            matrices_y = np.add(matrices_y,eyes_y)
           

            X = np.subtract(X,centers_x) # translate scene put origin on scene center
            X = np.matmul(X,matrices) # rotate
            X = np.add(X,centers_x) # translate back

            if not self.predict_offsets:
                  y = np.subtract(y,centers_y)
            y = np.matmul(y,matrices)

            if not self.predict_offsets:
                  y = np.add(y,centers_y)

            return X,y


      def __load_images(self):
            images = {}
            for scene in self.scene_list:
                  img = torch.FloatTensor(cv2.imread(self.images_path.format(scene)))
                  img = img.permute(2,0,1)
                  images[scene] = img
            return images
      def __get_matrices(self):

            matrices = {}

            for theta in [0] + self.augmentation_angles:
                  theta_rad = np.radians(theta)
                  c, s = np.cos(theta_rad), np.sin(theta_rad)
            
                  r = np.array([[c,-s],
                              [s,c]
                              ])
                  matrices[theta] = r 
            return matrices


# """
#       set_type:  train eval  test
#       use_images: True False
#       use_neighbors: True False
#       predict_offsets: 0: none, 1: based on last obs point, 2: based on previous point

#       data_type: frames trajectories
# """
# class Hdf5Dataset():
#       'Characterizes a dataset for PyTorch'
#       def __init__(self,images_path,hdf5_file,scene_list,t_obs,t_pred,set_type,use_images,data_type,use_neighbors_sample,use_neighbors_label,augmentation,augmentation_angles,centers,reduce_batches = True,predict_offsets = 0,predict_smooth=0,smooth_suffix = ""):

#             self.images_path = images_path + "{}.jpg"
#             self.hdf5_file = hdf5_file
#             self.set_type = set_type
#             self.scene_list = scene_list

#             self.data_type = data_type
#             self.use_images = use_images
#             self.use_neighbors_sample = use_neighbors_sample
#             self.use_neighbors_label = use_neighbors_label

            
#             self.centers = centers
#             self.reduce_batches = reduce_batches
#             self.predict_offsets = predict_offsets

#             self.predict_smooth = predict_smooth
#             self.smooth_suffix = smooth_suffix


#             self.dset_name = "samples_{}_{}".format(set_type,data_type)
#             self.dset_img_name = "images_{}_{}".format(set_type,data_type)
#             self.dset_types = "types_{}_{}".format(set_type,data_type)

#             self.t_obs = t_obs
#             self.t_pred = t_pred
#             self.seq_len = t_obs + t_pred
#             self.augmentation = augmentation 
#             self.augmentation_angles = augmentation_angles
            

#             if self.augmentation:
#                   self.r_matrices = self.__get_matrices()
#                   self.scene_list = helpers.helpers_training.augment_scene_list(self.scene_list,self.augmentation_angles)
            
#             self.images = self.__load_images()

#       def get_len(self):
#             with h5py.File(self.hdf5_file,"r") as hdf5_file: 
#                   if self.augmentation:
#                         return hdf5_file[self.dset_name].shape[0]  * (len(self.augmentation_angles) + 1)
#                   return hdf5_file[self.dset_name].shape[0]


#       def get_ids(self,ids):
#             with h5py.File(self.hdf5_file,"r") as hdf5_file: 
#                   coord_dset = hdf5_file[self.dset_name]
#                   scenes_dset = hdf5_file[self.dset_img_name]   
#                   types_dset = hdf5_file[self.dset_types]   

#                   if self.augmentation:

#                         red_ids = sorted(np.array(ids) % coord_dset.shape[0])
#                         m_ids = (np.array(ids) / float(coord_dset.shape[0]) ).astype(int)*90
#                         ids,matrix_indexes = [],[]

#                         for i in range(len(red_ids)):                              
#                               if i > 0 and  red_ids[i] == ids[-1]:
#                                     ids.append(red_ids[i]+1)
#                               else:
#                                     ids.append(red_ids[i])
            
#                   X,y,scenes = [],[],[]
#                   max_batch = coord_dset.shape[1]

#                   if self.reduce_batches:
#                         b,n,s,i = coord_dset.shape
#                         nb_agents = np.sum( np.sum(np.sum(coord_dset[ids,:,:self.t_obs],axis = 3),axis = 2) > 0, axis = 1 )                        
#                         max_batch = np.max(nb_agents)
                        

                        
#                   last_points = np.zeros((len(ids),max_batch,self.t_pred,2))


#                   if self.use_neighbors_sample:
#                         X = coord_dset[ids,:max_batch,:self.t_obs]


#                         if self.predict_offsets == 1:
#                               last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2)#B,N,tpred,2

#                   else: 
#                         # X = coord_dset[ids,0,:self.t_obs]
#                         X = np.expand_dims( coord_dset[ids,0,:self.t_obs] ,1)

#                         if self.predict_offsets == 1 :
#                               # last_points = np.repeat(  np.expand_dims(np.expand_dims(X,1)[:,:,-1],2),  self.t_pred, axis=2) #B,1,tpred,2
#                               last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2) #B,1,tpred,2

#                   if self.predict_smooth:
#                         coord_dset = hdf5_file[self.dset_name+self.smooth_suffix]

                        
#                   if self.use_neighbors_label:     
#                         y = coord_dset[ids,:max_batch,self.t_obs:self.seq_len] #B,N,tpred,2
#                         types = types_dset[ids,:max_batch] #B,N,tpred,2



                       
#                         if self.predict_offsets:

#                               if self.predict_offsets == 2:# y shifted left
#                                     last_points = coord_dset[ids,:max_batch,self.t_obs-1:self.seq_len-1]

#                               flat_y = y.reshape(-1,2)
#                               flat_last_points = last_points.reshape(-1,2)

#                               ids_ = np.argwhere(np.sum(flat_y,axis=1) > 0)
#                               flat_y[ids_] = np.subtract(flat_y[ids_],flat_last_points[ids_])

#                               y = flat_y.reshape(y.shape)

#                               # y = np.subtract(y,last_points)


                        
                              
#                   else:                        
#                         y = np.expand_dims( coord_dset[ids,0,self.t_obs:self.seq_len], 1) #B,1,tpred,2
                        
#                         types =  types_dset[ids,0] #B,1,tpred,2

#                         if self.predict_offsets:
#                               if self.use_neighbors_sample and self.predict_offsets == 1:
#                                     last_points = np.expand_dims( last_points[:,0], 1) #B,1,tpred,2

#                               if self.predict_offsets == 2: # y shifted left
#                                     last_points = np.expand_dims( coord_dset[ids,0,self.t_obs-1:self.seq_len-1], 1)

#                               flat_y = y.reshape(-1,2)
#                               flat_last_points = last_points.reshape(-1,2)

#                               ids_ = np.argwhere(np.sum(flat_y,axis=1) > 0)
#                               flat_y[ids_] = np.subtract(flat_y[ids_],flat_last_points[ids_])

#                               y = flat_y.reshape(y.shape)    
#                               y = np.subtract(y,last_points)


#                   scenes = [img.decode('UTF-8') for img in scenes_dset[ids]] # B

                 

#                   if self.augmentation:
#                         centers = np.array([self.centers[scene] for scene in scenes]) # B,2
#                         centers = np.expand_dims(centers,axis = 1) # B,1,2
#                         centers = np.expand_dims(centers,axis = 1) # B,1,1,2
#                         centers = np.repeat(centers,X.shape[1],axis = 1) # B,N,1,2

#                         centers_x = np.repeat(centers,X.shape[2],axis = 2) # B,N,t_obs,2
#                         centers_y = np.repeat(centers,y.shape[2],axis = 2) # B,N,t_pred,2

#                         matrices = np.array([self.r_matrices[m] for m in m_ids])
#                         matrices = np.expand_dims(matrices,axis = 1)
#                         matrices = np.repeat(matrices,X.shape[1],axis = 1)
                        
#                         if self.use_neighbors_sample:
#                               X[:,(nb_agents-1)] = np.subtract(X[:,(nb_agents-1)],centers_x[:,(nb_agents-1)]) #center x

#                               if not self.predict_offsets:
#                                     y[:,(nb_agents-1)] = np.subtract(y[:,(nb_agents-1)],centers_y[:,(nb_agents-1)]) #center y if not offsets 
#                                     # excluding padding [0,0] points
#                               X[:,(nb_agents-1)] = np.matmul(X[:,(nb_agents-1)],matrices[:,(nb_agents-1)]) #rotate x around 0,0
#                               y[:,(nb_agents-1)] = np.matmul(y[:,(nb_agents-1)],matrices[:,(nb_agents-1)]) #rotate y around 0,0

#                               X[:,(nb_agents-1)] = np.add(X[:,(nb_agents-1)],centers_x[:,(nb_agents-1)]) #translate back

#                               if not self.predict_offsets:
#                                     y[:,(nb_agents-1)] = np.add(y[:,(nb_agents-1)],centers_y[:,(nb_agents-1)]) #translate back

#                         else:
#                               X = np.subtract(X,centers_x)
#                               if not self.predict_offsets:
#                                     y = np.subtract(y,centers_y)
#                               X = np.matmul(X,matrices)
#                               y = np.matmul(y,matrices)

#                               X = np.add(X,centers_x)
#                               if not self.predict_offsets:
#                                     y = np.add(y,centers_y)
#                         scenes = [scene if m == 0 else scene +"_{}".format(m) for scene,m in zip(scenes,m_ids)] # B
                               

                  





#                   if not self.use_images:
#                         return (torch.FloatTensor(X).contiguous(),torch.FloatTensor(y).contiguous(),scenes,torch.FloatTensor(types))
            
#                   imgs = torch.stack([self.images[img] for img in scenes],dim = 0) 
                  
#                   return (torch.FloatTensor(X).contiguous(),torch.FloatTensor(y).contiguous(),imgs,scenes,torch.FloatTensor(types))


#       def __load_images(self):
#             images = {}
#             for scene in self.scene_list:
#                   img = torch.FloatTensor(cv2.imread(self.images_path.format(scene)))
#                   img = img.permute(2,0,1)
#                   images[scene] = img
#             return images
#       def __get_matrices(self):

#             matrices = {}

#             for theta in [0] + self.augmentation_angles:
#                   theta_rad = np.radians(theta)
#                   c, s = np.cos(theta_rad), np.sin(theta_rad)
            
#                   r = np.array([[c,-s],
#                               [s,c]
#                               ])
#                   matrices[theta] = r 
#             return matrices


        


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
