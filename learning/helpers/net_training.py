import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import helpers.helpers_training as helpers
import os


"""
    Train loop for an epoch
    Uses cuda if available
    LOss is averaged for a batch
    THen averaged batch losses are averaged
    over the number of batches
"""
def train(model, device, train_loader,criterion, optimizer, epoch,batch_size,print_every = 10):
    model.train()
    epoch_loss = 0.
    batches_loss = []
    torch.cuda.synchronize()

    start_time = time.time()

    nb_grad_plots = 200
    ids_grads = np.arange(int(train_loader.nb_batches) )
    np.random.shuffle(ids_grads)
    ids_grads = ids_grads[:nb_grad_plots]
    # print(ids_grads)

    for batch_idx, data in enumerate(train_loader):
        s = time.time()
        inputs, labels, ids,types,points_mask, active_mask, imgs = data
        inputs, labels,types, imgs = inputs.to(device), labels.to(device), types.to(device) , imgs.to(device)
        active_mask = active_mask.to(device)


        # torch.cuda.synchronize()
        # print("data loading {}".format(time.time()-s))
        # s = time.time()

        
        optimizer.zero_grad()
        outputs = model((inputs,types,active_mask,points_mask,imgs))

        # torch.cuda.synchronize()
        # print("overall model {}".format(time.time()-s))
        # s = time.time()
# ####################
     

        # torch.cuda.synchronize()
        # print("masking {}".format(time.time()-s))
        # s = time.time()
# ###########################
        points_mask = torch.FloatTensor(points_mask).to(device)
        outputs = torch.mul(points_mask,outputs)
        labels = torch.mul(points_mask,labels)

        # if there is some padding at the end of the trajectory, we train to predict it
        # we don't count it at test time
        mask_loss = (torch.sum(torch.sum(points_mask,dim = 3),dim = 2) > 0).unsqueeze(2).repeat(1,1,labels.size()[2])
        mask_loss = mask_loss.unsqueeze(3).repeat(1,1,1,2).type(torch.FloatTensor).to(device)
        loss = criterion(outputs, labels,mask_loss)
        # print(loss)


        # torch.cuda.synchronize()
        # print("loss {}".format(time.time()-s))
        # s = time.time()

        loss.backward()

        # torch.cuda.synchronize()
        # print("backward {}".format(time.time()-s))
        
        # s = time.time()

        if batch_idx in ids_grads:
            helpers.plot_params(model.named_parameters(),epoch)

            helpers.plot_grad_flow(model.named_parameters(),epoch)

        # torch.cuda.synchronize()
        # print("grad flow {}".format(time.time()-s))
        
        # s = time.time()
        optimizer.step()

        epoch_loss += loss.item()
        batches_loss.append(loss.item())

        if batch_idx % print_every == 0:
            print(batch_idx,loss.item(),time.time()-start_time)  
            # print(batch_idx,time.time()-start_time)  
            # print(time.time()-start_time)
            
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
def evaluate(model, device, eval_loader,criterion, epoch, batch_size,scalers_path,multiple_scalers,
            model_type,nb_plots = 8, offsets = 0 ,normalized = 0):
    model.eval()
    eval_loss = 0.
    fde = 0.
    ade = 0.
    eval_loader_len =   float(eval_loader.nb_batches)
    nb_sample = eval_loader_len*batch_size
    
    start_time = time.time()


    nb_batches = eval_loader.nb_batches
    kept_batches_id = np.arange(nb_batches)
    np.random.shuffle(kept_batches_id)

    kept_batches_id = kept_batches_id[:nb_plots]

    kept_samples = []
    for i,data in enumerate(eval_loader):
        keep_batch = (i in kept_batches_id )


        inputs, labels, ids,types,points_mask, active_mask,img = data
        inputs, labels,types , img = inputs.to(device), labels.to(device), types.to(device),img.to(device)
        active_mask = active_mask.to(device)
        
        s = time.time()
        # outputs = model((inputs,types))
        outputs = model((inputs,types,active_mask,points_mask,img))


        #### function takes inputs labels outputs offsets ###
        #### returns labels and outputs in trajectory format ####

        # inv_labels,inv_outputs = labels,outputs


        
        if normalized:
            # if model_type == 0:
            #     inv_labels,inv_outputs = helpers.revert_scaling(ids,labels,outputs,scalers_path,multiple_scalers)
            #     inv_outputs = inv_outputs.view(inv_labels.size())
            # elif model_type == 1:
            #     inv_labels,inv_outputs = helpers.revert_scaling(ids,labels,outputs[:,:,:2],scalers_path,multiple_scalers)
            
            
            _,_,inputs = helpers.revert_scaling(ids,labels,outputs,inputs,scalers_path,multiple_scalers)
            # labels,outputs,inputs = helpers.revert_scaling(ids,labels,outputs,inputs,scalers_path,multiple_scalers)

            # labels,outputs = helpers.revert_scaling(ids,labels,outputs,scalers_path,multiple_scalers)

            outputs = outputs.view(labels.size())

            inputs,labels,outputs = helpers.offsets_to_trajectories(inputs.detach().cpu().numpy(),
                                                                labels.detach().cpu().numpy(),
                                                                outputs.detach().cpu().numpy(),
                                                                offsets)
                                                                
        
        


        

        inputs,labels,outputs = torch.FloatTensor(inputs).to(device),torch.FloatTensor(labels).to(device),torch.FloatTensor(outputs).to(device)
        

        # we don't count the prediction error for end of trajectory padding
        points_mask = torch.FloatTensor(points_mask).to(device)#
        outputs = torch.mul(points_mask,outputs)#
        labels = torch.mul(points_mask,labels)#


        loss = criterion(outputs, labels,points_mask)

        if keep_batch:
            kept_sample_id = np.random.randint(0,labels.size()[0])

            l = labels[kept_sample_id]
            o = outputs[kept_sample_id,:,:,:2]
            ins = inputs[kept_sample_id]
            if model_type == 0: ####
                ins = inputs[kept_sample_id].unsqueeze(0)
            elif model_type == 1:

                sample_mask = points_mask[kept_sample_id]
                # kept_mask = helpers.mask_loss(l.unsqueeze(0).detach().cpu().numpy())
                kept_mask = helpers.mask_loss(sample_mask.detach().cpu().numpy())


            

                l = l[kept_mask]
                o = o[kept_mask]
                ins = ins[kept_mask]





            kept_samples.append((
                ins.detach().cpu().numpy(),
                l.detach().cpu().numpy(),
                o.detach().cpu().numpy()
                ))
        

####################
        # mask = helpers.mask_loss(labels.detach().cpu().numpy())
        # outputs = outputs.contiguous().view([outputs.size()[0] * outputs.size()[1]] + list(outputs.size()[2:]))

        # labels = labels.contiguous().view([labels.size()[0] * labels.size()[1]] + list(labels.size()[2:]))

        # outputs = outputs[mask]
        # labels = labels[mask]
###########################
        


###########################
        
        
        

        ade += helpers.ade_loss(outputs,labels,points_mask).item() ######
        fde += helpers.fde_loss(outputs,labels,points_mask).item()
      
        eval_loss += loss.item()

    # print(len(kept_samples))
    helpers.plot_samples(kept_samples,epoch,1,1) #### retrieve 

            
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
def training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,
        scalers_path,multiple_scalers,model_type,plot = True,early_stopping = True,load_path = None,plot_every = 5,
        save_every = 1,offsets = 0,normalized = 0):

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
    
    # try:
    for epoch in range(start_epoch,n_epochs):
        train_loss,_ = train(net, device, train_loader,criterion_train, optimizer, epoch,batch_size)
        
        # train_loss,fde,ade = evaluate(net, device, train_loader,criterion_eval, 
        #         epoch, batch_size,scalers_path,multiple_scalers,model_type,offsets=offsets,
        #         normalized =normalized )
        
        eval_loss,fde,ade = evaluate(net, device, eval_loader,criterion_eval, 
                epoch, batch_size,scalers_path,multiple_scalers,model_type,offsets=offsets,
                normalized =normalized )
            

        losses["train"]["loss"].append(train_loss)
        losses["eval"]["loss"].append(eval_loss)
        losses["eval"]["ade"].append(ade)
        losses["eval"]["fde"].append(fde)


        if plot and epoch % plot_every == 0:
            plot_losses(losses,s,root = "./data/reports/losses/")

        if epoch % save_every == 0:
            save_model(epoch,net,optimizer,losses)

        print(time.time()-s)
        
    
    # except Exception as e: 
    #     print(e)

    save_model(epoch+1,net,optimizer,losses)
    if plot:
        plot_losses(losses,s,root = "./data/reports/losses/")
     
    return losses,net


def plot_losses(losses,idx,root = "./data/reports/losses/"):
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

    dirs = os.listdir(save_root)

    save_path = save_root + "model_{}_{}.tar".format(epoch,time.time())


    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),             
        'losses': losses
        }
    # state = {
    #     'state_dict': net.state_dict(),
    #     }
    torch.save(state, save_path)

    for dir_ in dirs:
        os.remove(save_root+dir_)
    
    print("model saved in {}".format(save_path))







