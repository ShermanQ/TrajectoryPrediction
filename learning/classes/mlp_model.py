
import torch
import torch.nn as nn
import torch.nn.functional as f

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.fc5 = nn.Linear(hidden_size,hidden_size)

        self.output = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)
        x = f.relu(x)
        x = self.fc4(x)
        x = f.relu(x)
        x = self.fc5(x)
        x = f.relu(x)
        
        x = self.output(x)
        return x