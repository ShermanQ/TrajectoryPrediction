import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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

    def forward(self,x):
        encoder_output, encoder_state = self.encoder(x)

    
        output,state = encoder_output,encoder_state
        # output = x[:,-1].view(200,1,2) # take last pos of input sequence as <sos>

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

def train(model, device, train_loader,criterion, optimizer, epoch,batch_size,print_every = 100):
    model.train()
    epoch_loss = 0.
    batches_loss = []

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches_loss.append(loss.item())

        if batch_idx % print_every == 0:
            print(batch_idx,loss.item(),time.time()-start_time)       
    epoch_loss /= float(len(train_loader))        
    print('Epoch n {} Loss: {}'.format(epoch,epoch_loss))

    return epoch_loss,batches_loss

def evaluate(model, device, eval_loader,criterion, epoch, batch_size):
    model.eval()
    eval_loss = 0.
    nb_sample = len(eval_loader)*batch_size
    
    start_time = time.time()
    for data in eval_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        
        eval_loss += loss.item()

             
    eval_loss /= float(len(eval_loader))        
    print('Epoch n {} Evaluation Loss: {}'.format(epoch,eval_loss))

    return eval_loss

def save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,save_root = "./learning/data/models/" ):

    save_path = save_root + "model_{}.pt".format(time.time())


    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),             
        'train_losses': train_losses, 
        'train_losses': eval_losses, 
        'eval_losses': train_losses,
        'batch_losses': batch_losses 
        }
    torch.save(state, save_path)
    
    print("model saved in {}".format(save_path))
def training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,plot = True,early_stopping = True):

    train_losses = []
    eval_losses = []

    batch_losses = []


    s = time.time()
    
    try:
        for epoch in range(n_epochs):
            train_loss,batches_loss = train(net, device, train_loader,criterion_train, optimizer, epoch,batch_size)
            batch_losses += batches_loss
            train_losses.append(train_loss)

            eval_loss = evaluate(net, device, eval_loader,criterion_eval, epoch, batch_size)
        
            eval_losses.append(eval_loss)
            print(time.time()-s)
        
    except :
        # logging.error(traceback.format_exc())
        # save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,save_path)
        pass

    save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses)
    if plot:
        plt.plot(train_losses)
        plt.plot(eval_losses)
        plt.show()

    return train_losses,eval_losses,batch_losses

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
    # data_path = "./data/deep/data.csv"
    # label_path = "./data/deep/labels.csv"

    # samples_path = "./learning/data/samples/"
    # labels_path = "./learning/data/labels/"

    
    # extract_tensors(data_path,label_path,samples_path,labels_path)
    # print(time.time()-s)
       
    # set pytorch
    torch.manual_seed(10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    # parameters
    input_size,hidden_size,num_layers,output_size,batch_size,seq_len,seq_len_d  = 2,30,3,2,200,8,12

    nb_samples = 42380
    learning_rate = 0.001
    n_epochs = 10

    # split train eval indices
    train_indices,eval_indices = train_test_split([i for i in range(nb_samples)],test_size = 0.2,random_state = 42)

    # load datasets
    train_dataset = CustomDataset(train_indices,"./learning/data/")
    eval_dataset = CustomDataset(eval_indices,"./learning/data/")

    # create dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size= batch_size, shuffle=True,num_workers= 10,drop_last = True)
    eval_loader = torch.utils.data.DataLoader( eval_dataset, batch_size= batch_size, shuffle=False,num_workers= 10,drop_last = True)

    # init model and send it to gpu
    net = seq2seq(input_size,output_size,hidden_size,hidden_size,num_layers,num_layers,batch_size,seq_len_d)
    net.to(device)
    
    #losses
    criterion_train = nn.MSELoss()
    criterion_eval = nn.MSELoss(reduction="sum")
    criterion_eval = criterion_train

    #optimizer
    optimizer = optim.Adam(net.parameters(),lr = 0.001)

    
    train_losses,eval_losses,batch_losses = training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer)
    

if __name__ == "__main__":
    main()




