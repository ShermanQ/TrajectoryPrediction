import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def custom_loss(outputs,labels):
    mse = nn.MSELoss(reduction = "none")
    loss = mse(outputs, labels)
    loss = torch.sum(loss,dim = 2 , keepdim = False)
    loss = torch.mean(loss, dim = 1)
    # loss = torch.sum(loss, dim = 1)

    loss = torch.mean(loss,dim = 0)
    return loss
"""
    Train loop for an epoch
    Uses cuda if available
    LOss is averaged for a batch
    THen averaged batch losses are averaged
    over the number of batches
"""
def train(model, device, train_loader,criterion, optimizer, epoch,batch_size,print_every = 100):
    model.train()
    epoch_loss = 0.
    batches_loss = []

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs,labels)

        # print("---------")
        # print(output.size())
        # print(labels.size())
        loss = criterion(output, labels)
        # loss = custom_loss(output, labels)



        # output = model(inputs)
        

        # print(loss.size())
        # print(output[0])
        # print(labels[0])
        # print(loss)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches_loss.append(loss.item())

        if batch_idx % print_every == 0:
            print(batch_idx,loss.item(),time.time()-start_time)       
    epoch_loss /= float(len(train_loader))        
    print('Epoch n {} Loss: {}'.format(epoch,epoch_loss))

    return epoch_loss,batches_loss

"""
    Evaluation loop for an epoch
    Uses cuda if available
    LOss is averaged for a batch
    THen averaged batch losses are averaged
    over the number of batches

    FDE loss is added using MSEerror on the last point of prediction and target
    sequences
"""
def evaluate(model, device, eval_loader,criterion, epoch, batch_size):
    model.eval()
    eval_loss = 0.
    fde_loss = 0.
    nb_sample = len(eval_loader)*batch_size
    
    start_time = time.time()
    for data in eval_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs,labels)
        print("---------")
        # print(output.size())
        # print(labels.size())

        loss = criterion(output, labels)
        # print(loss.size())

        
        # print(inputs[-10:])
        print(labels[-10:])
        print(output[-10:])
        print(labels[:10])
        print(output[:10])
        # print(loss.item())
        # print("---------")

        # print(output[:,-1].view(200,1,2))
        # print(labels.size())
        fde_loss += criterion(output[:,-1].view(batch_size,1,-1), labels[:,-1].view(batch_size,1,-1)).item()
        
        eval_loss += loss.item()

             
    eval_loss /= float(len(eval_loader))        
    fde_loss /= float(len(eval_loader))        
    print('Epoch n {} Evaluation Loss: {}, FDE Loss {}'.format(epoch,eval_loss,fde_loss))

    return eval_loss,fde_loss


"""
    Saves model and optimizer states as dict
    THe current epoch is stored
    THe different losses at previous time_steps are loaded

"""
def save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,fde_losses,save_root = "./learning/data/models/" ):

    save_path = save_root + "model_{}.tar".format(time.time())


    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),             
        'train_losses': train_losses,  
        'eval_losses': eval_losses,
        'batch_losses': batch_losses,
        'fde_losses': fde_losses
        }
    torch.save(state, save_path)
    
    print("model saved in {}".format(save_path))

"""
    Training loop
    For NUMBER OF EPOCHS calls train and evaluate
    if a model path is given, loads model
    and resume training
    if plot, display the different losses
    If exception during training, model is stored
"""
def training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,plot = True,early_stopping = True,load_path = None):

    train_losses = []
    eval_losses = []
    batch_losses = []
    fde_losses = []
    start_epoch = 0


    if load_path is not None:
        print("loading former model from {}".format(load_path))
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_losses = checkpoint["train_losses"]
        eval_losses = checkpoint["eval_losses"]
        batch_losses = checkpoint["batch_losses"]
        fde_losses = checkpoint["fde_losses"]
        start_epoch = checkpoint["epoch"]


    s = time.time()
    
    try:
        for epoch in range(start_epoch,n_epochs):
            train_loss,batches_loss = train(net, device, train_loader,criterion_train, optimizer, epoch,batch_size)
            batch_losses += batches_loss
            train_losses.append(train_loss)

            temp = net.teacher_forcing
            net.teacher_forcing = False
            eval_loss,fde_loss = evaluate(net, device, eval_loader,criterion_eval, epoch, batch_size)
            net.teacher_forcing = temp

            eval_losses.append(eval_loss)
            fde_losses.append(fde_loss)
            print(time.time()-s)
        
    except :
        # logging.error(traceback.format_exc())
        # save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,save_path)
        pass

    save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,fde_losses)
    if plot:
        plt.plot(train_losses)
        plt.plot(eval_losses)
        plt.plot(fde_losses)
        plt.show()

    return train_losses,eval_losses,batch_losses