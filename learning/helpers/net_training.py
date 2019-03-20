import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import helpers.helpers_training as helpers


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
        inputs, labels, ids = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
# ####################
        mask = helpers.mask_loss(labels.detach().cpu().numpy())
        outputs = outputs.contiguous().view([outputs.size()[0] * outputs.size()[1]] + list(outputs.size()[2:]))

        labels = labels.contiguous().view([labels.size()[0] * labels.size()[1]] + list(labels.size()[2:]))

        outputs = outputs[mask]
        labels = labels[mask]
# ###########################
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches_loss.append(loss.item())

        if batch_idx % print_every == 0:
            # print(batch_idx,loss.item(),time.time()-start_time)  
            print(batch_idx,time.time()-start_time)  
            
    # epoch_loss /= float(len(train_loader))   
    epoch_loss /= float(train_loader.nb_batches)        

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

    model: 0 rnn_mlp
           1 iatcnn
"""
def evaluate(model, device, eval_loader,criterion, epoch, batch_size,scalers_path,multiple_scalers,model_type ):
    model.eval()
    eval_loss = 0.
    fde = 0.
    ade = 0.
    eval_loader_len =   float(eval_loader.nb_batches)
    nb_sample = eval_loader_len*batch_size
    
    start_time = time.time()
    for data in eval_loader:
        inputs, labels, ids = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        # output = output.view(labels.size())

####################
        mask = helpers.mask_loss(labels.detach().cpu().numpy())
        outputs = outputs.contiguous().view([outputs.size()[0] * outputs.size()[1]] + list(outputs.size()[2:]))

        labels = labels.contiguous().view([labels.size()[0] * labels.size()[1]] + list(labels.size()[2:]))

        outputs = outputs[mask]
        labels = labels[mask]
###########################
        loss = criterion(outputs, labels)
        inv_labels,inv_outputs = None,None
        if model_type == 0:
            inv_labels,inv_outputs = helpers.revert_scaling(ids,labels,outputs,scalers_path,multiple_scalers)
            inv_outputs = inv_outputs.view(inv_labels.size())
        elif model_type == 1:
            inv_labels,inv_outputs = helpers.revert_scaling(ids,labels,outputs[:,:,:2],scalers_path,multiple_scalers)

        

        ade += helpers.ade_loss(inv_outputs,inv_labels).item()
        fde += helpers.fde_loss(inv_outputs,inv_labels).item()
      
        eval_loss += loss.item()

            
    eval_loss /= eval_loader_len 
    ade /= eval_loader_len      
    fde /= eval_loader_len        

    print('Epoch n {} Evaluation Loss: {}, ADE: {}, FDE: {}'.format(epoch,eval_loss,ade,fde))


    return eval_loss,fde,ade

"""
    Training loop
    For NUMBER OF EPOCHS calls train and evaluate
    if a model path is given, loads model
    and resume training
    if plot, display the different losses
    If exception during training, model is stored
"""
def training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,scalers_path,multiple_scalers,model_type,plot = True,early_stopping = True,load_path = None,plot_every = 5):

    losses = {
        "train":{
            "loss": []

        },
        "eval":{
            "loss": [],
            "fde":[],
            "ade":[]

        }
    }

    start_epoch = 0


    if load_path != "":
        print("loading former model from {}".format(load_path))
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses = checkpoint["losses"]
        start_epoch = checkpoint["epoch"]


    s = time.time()
    
    try:
        for epoch in range(start_epoch,n_epochs):
            train_loss,_ = train(net, device, train_loader,criterion_train, optimizer, epoch,batch_size)
           
            
            eval_loss,fde,ade = evaluate(net, device, eval_loader,criterion_eval, epoch, batch_size,scalers_path,multiple_scalers,model_type)
            

            losses["train"]["loss"].append(train_loss)
            losses["eval"]["loss"].append(eval_loss)
            losses["eval"]["ade"].append(ade)
            losses["eval"]["fde"].append(fde)


            if plot and epoch % plot_every == 0:
                plot_losses(losses,s,root = "./data/reports/")

            print(time.time()-s)
        
    except :
        # logging.error(traceback.format_exc())
        # save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,save_path)
        pass

    save_model(epoch,net,optimizer,losses)
    if plot:
        plot_losses(losses,s,root = "./data/reports/")
     
    return losses


def plot_losses(losses,idx,root = "./data/reports/"):
    plt.plot(losses["train"]["loss"],label = "train_loss")
    plt.plot(losses["eval"]["loss"],label = "eval_loss")
    plt.legend()

    # plt.show()
    plt.savefig("{}losses_{}.jpg".format(root,idx))
    plt.close()

    plt.plot(losses["eval"]["ade"],label = "ade")
    plt.plot(losses["eval"]["fde"],label = "fde")
    plt.legend()

    plt.savefig("{}ade_fde_{}.jpg".format(root,idx))
    plt.close()




"""
    Saves model and optimizer states as dict
    THe current epoch is stored
    THe different losses at previous time_steps are loaded

"""
def save_model(epoch,net,optimizer,losses,save_root = "./learning/data/models/" ):

    save_path = save_root + "model_{}.tar".format(time.time())


    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),             
        'losses': losses
        }
    torch.save(state, save_path)
    
    print("model saved in {}".format(save_path))







