import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
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
    
def main():
    
    
    # torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader( trainset, batch_size= 200 , shuffle=True)
        



    


    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    input_size,hidden_size,output_size = 28*28,1000,10
    net = MLP(input_size,hidden_size,output_size)

    net.to(device)

    learning_rate = 0.001
    n_epochs = 10
    #loss
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.Adam(net.parameters(),lr = 0.001)

    print(torch.cuda.is_available())
    losses = []

    # for epoch in range(1, n_epochs + 1):
    #     loss = train(net, device, train_loader,criterion, optimizer, epoch)
    #     losses.append(loss)
    for epoch in range(n_epochs):
        epoch_loss = 0.
        print(epoch)
        for i, data in enumerate(train_loader):

            inputs,targets = data
            inputs = inputs.view(200,-1)
            inputs,targets = inputs.to(device),targets.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(epoch_loss)
        losses.append(epoch_loss)

    plt.plot(losses)


if __name__ == "__main__":
    main()