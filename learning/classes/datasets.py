import torch
from torch.utils import data
import cv2

"""
      Custom pytorch dataset
      self.list_IDs: ids of sample considered in this dataset
      self.data_path: path whereto read the data

      __getitem__: given an index, selects the corresponding sample id
      and load data and labels files 
      for now return only the trajectory of the main agent, not its neighbors

      It'S done this way so multiprocessing can be used when loading batch with pytorch dataloader
"""
class CustomDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs,data_path):
        'Initialization'
        self.list_IDs = list_IDs
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

        X = torch.load(self.data_path + "samples/sample" + str(ID) + '.pt')[0]
        y = torch.load(self.data_path + "labels/label" + str(ID) + '.pt')[0]
        
      #   print("x")
      #   print(X)
      #   print("y")
      #   print(y)
        return X, y


"""
      Custom pytorch dataset
      self.list_IDs: ids of sample considered in this dataset
      self.data_path: path whereto read the data

      __getitem__: given an index, selects the corresponding sample id
      and load data and labels files 
      for now return only the trajectory of the main agent, not its neighbors

      It'S done this way so multiprocessing can be used when loading batch with pytorch dataloader
"""
class CustomDatasetIATCNN(data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, list_IDs,data_path):
            'Initialization'
            self.list_IDs = list_IDs
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
            
            return X,img, y