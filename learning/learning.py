import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import csv
import numpy as np
import time

from classes import CustomDataset


class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,input_size,hidden_size,num_layers,batch_size):
        super(encoderLSTM,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.output_size = output_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden_state()

        self.lstm = nn.LSTM(input_size = self.input_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        self.last_output = nn.Linear(self.hidden_size,self.input_size)

        

    def forward(self,x):
        self.hidden = self.init_hidden_state()
        output,self.hidden = self.lstm(x,self.hidden)
        decoder_start = self.last_output(output[:,-1])
        return decoder_start.view(self.batch_size,1,self.input_size), self.hidden

    def init_hidden_state(self):
        h_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()

        return (h_0,c_0)

class decoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,output_size,hidden_size,num_layers,batch_size):
        super(decoderLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden_state()
        # self.seq_len = seq_len
        # self.target = target

        
        self.lstm = nn.LSTM(input_size = self.output_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        self.out = nn.Linear(self.hidden_size,self.output_size)

        

    def forward(self,x,hidden):
        # self.hidden = self.init_hidden_state()
        # x = x.view(self.seq_len,self.batch_size,2)
        output,self.hidden = self.lstm(x,hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden_state(self):
        h_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()

        return (h_0,c_0)

class seq2seq(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,hidden_size_d,num_layers,num_layers_d,batch_size,seq_len_d,teacher_forcing = False):
        super(seq2seq,self).__init__()
        self.encoder = encoderLSTM(input_size,hidden_size,num_layers,batch_size)
        self.decoder = decoderLSTM(output_size,hidden_size_d,num_layers_d,batch_size)
        self.seq_len_d = seq_len_d
        self.batch_size = batch_size
        self.output_size = output_size

    def forward(self,x,y):
        encoder_output, encoder_state = self.encoder(x)

    
        output,state = encoder_output,encoder_state
        outputs = []
        for _ in range(self.seq_len_d):
            output,state = self.decoder(output,state)
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs.view(self.batch_size,self.seq_len_d,self.output_size)


        

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

def train(model, device, train_loader,criterion, optimizer, epoch):
        model.train()
        epoch_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            data, target = data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        return epoch_loss

def extract_tensors(data_path,label_path,samples_path,labels_path):
    with open(data_path) as data_csv :
        with open(label_path) as label_csv:
            data_reader = csv.reader(data_csv)
            label_reader = csv.reader(label_csv)

            for data,label in zip(data_reader,label_reader):
                sample_id,nb_objects,t_obs,t_pred = data[0],int(data[1]),int(data[2]),int(data[3])
                features = data[4:]
                labels = label[1:]

                features = torch.FloatTensor([float(f) for f in features])
                features = features.view(nb_objects,t_obs,2)

                labels = torch.FloatTensor([float(f) for f in labels])
                labels = labels.view(nb_objects,t_pred,2)


                # features = np.array([float(f) for f in features])
                # features = features.reshape(nb_objects,t_obs,2)

                torch.save(features,samples_path+"sample"+sample_id+".pt")
                torch.save(labels,labels_path+"label"+sample_id+".pt")

# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     return data, target

def naive_collate_lstm(batch):
    data = [item[0][0] for item in batch]
    target = [item[1][0] for item in batch]
    return data, target

def main():
    
    nb_samples = 42380
    # nb_samples = 10000

    partition = [i for i in range(nb_samples)]
    dataset = CustomDataset(partition,"./learning/data/")
    # loader = torch.utils.data.DataLoader( dataset, batch_size= 200,collate_fn=naive_collate_lstm, shuffle=True)
    
    

    # data_path = "./data/deep/data.csv"
    # label_path = "./data/deep/labels.csv"

    # samples_path = "./learning/data/samples/"
    # labels_path = "./learning/data/labels/"

    
    # extract_tensors(data_path,label_path,samples_path,labels_path)
    # print(time.time()-s)
       
    torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

        



    
    

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    # input_size,hidden_size,output_size = 16,1000,24
    # net = MLP(input_size,hidden_size,output_size)
    input_size,hidden_size,num_layers,output_size,batch_size,seq_len,seq_len_d  = 2,30,1,2,200,8,12

    loader = torch.utils.data.DataLoader( dataset, batch_size= batch_size, shuffle=True,num_workers= 10,drop_last = True)

    # net = encoderLSTM(input_size,hidden_size,num_layers,batch_size)
    # net.to(device)
    # decoder = decoderLSTM(output_size,hidden_size,num_layers,batch_size)
    # decoder.to(device)

    net = seq2seq(input_size,output_size,hidden_size,hidden_size,num_layers,num_layers,batch_size,seq_len_d)
    net.to(device)
    learning_rate = 0.001
    n_epochs = 1
    #loss
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.Adam(net.parameters(),lr = 0.001)

    print(torch.cuda.is_available())
    losses = []


    s = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.
        print(epoch)
        print(time.time()-s)

        for i, data in enumerate(loader):
            
            

            inputs,targets = data

            inputs,targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            
            outputs = net(inputs,targets)
            print(outputs)
            print(len(outputs))

            print(outputs.size())

            print(targets.size())
            # start,hidden_state = net(inputs)
            # print(start.size())
            # output, hidden = decoder(start,hidden_state)
            # print(output.size())
            # loss = criterion(outputs,targets)

    #         # # del inputs
    #         # # del targets

    #         loss.backward()
    #         optimizer.step()

    #         # epoch_loss += loss.item()/200

            if i % 50 == 0:
                print(i)       
                # print(loss.item()/200)
                print(time.time()-s)
            break
        break            

        # print(epoch_loss/i)
        # losses.append(epoch_loss)

    # plt.plot(losses)


if __name__ == "__main__":
    main()