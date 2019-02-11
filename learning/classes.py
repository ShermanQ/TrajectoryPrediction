import torch
from torch.utils import data

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
        
        return X, y