import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load

"""
    Train loop for an epoch
    Uses cuda if available
    LOss is averaged for a batch
    THen averaged batch losses are averaged
    over the number of batches
"""
def train(model, device, train_loader,criterion, optimizer, epoch,batch_size,print_every = 1):
    model.train()
    epoch_loss = 0.
    batches_loss = []

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels, ids = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
####################
        mask = mask_loss(labels.detach().cpu().numpy())
        outputs = outputs.contiguous().view([outputs.size()[0] * outputs.size()[1]] + list(outputs.size()[2:]))

        labels = labels.contiguous().view([labels.size()[0] * labels.size()[1]] + list(labels.size()[2:]))

        outputs = outputs[mask]
        labels = labels[mask]
###########################
        loss = criterion(outputs, labels)
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

    model: 0 rnn_mlp
           1 iatcnn
"""
def evaluate(model, device, eval_loader,criterion, epoch, batch_size,scalers_path,multiple_scalers,model_type ):
    model.eval()
    eval_loss = 0.
    fde = 0.
    ade = 0.

    nb_sample = len(eval_loader)*batch_size
    
    start_time = time.time()
    for data in eval_loader:
        inputs, labels, ids = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        # output = output.view(labels.size())

####################
        mask = mask_loss(labels.detach().cpu().numpy())
        outputs = outputs.contiguous().view([outputs.size()[0] * outputs.size()[1]] + list(outputs.size()[2:]))

        labels = labels.contiguous().view([labels.size()[0] * labels.size()[1]] + list(labels.size()[2:]))

        outputs = outputs[mask]
        labels = labels[mask]
###########################
        loss = criterion(outputs, labels)
        inv_labels,inv_outputs = None,None
        if model_type == 0:
            inv_labels,inv_outputs = revert_scaling(ids,labels,outputs,scalers_path,multiple_scalers)
            inv_outputs = inv_outputs.view(inv_labels.size())
        elif model_type == 1:
            inv_labels,inv_outputs = revert_scaling(ids,labels,outputs[:,:,:2],scalers_path,multiple_scalers)

        

        ade += ade_loss(inv_outputs,inv_labels).item()
        fde += fde_loss(inv_outputs,inv_labels).item()
      
        eval_loss += loss.item()

             
    eval_loss /= float(len(eval_loader))  
    ade /= float(len(eval_loader))        
    fde /= float(len(eval_loader))        

    print('Epoch n {} Evaluation Loss: {}, ADE: {}, FDE: {}'.format(epoch,eval_loss,ade,fde))


    return eval_loss,fde,ade


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
def training_loop(n_epochs,batch_size,net,device,train_loader,eval_loader,criterion_train,criterion_eval,optimizer,scalers_path,multiple_scalers,model_type,plot = True,early_stopping = True,load_path = None):

    train_losses = []
    eval_losses = []
    batch_losses = []
    fde_losses = []
    ade_losses = []

    start_epoch = 0


    if load_path != "":
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
    
    # try:
    for epoch in range(start_epoch,n_epochs):
        train_loss,batches_loss = train(net, device, train_loader,criterion_train, optimizer, epoch,batch_size)
        batch_losses += batches_loss
        train_losses.append(train_loss)

        
        eval_loss,fde,ade = evaluate(net, device, eval_loader,criterion_eval, epoch, batch_size,scalers_path,multiple_scalers,model_type)
        

        eval_losses.append(eval_loss)
        fde_losses.append(fde)
        ade_losses.append(ade)

        print(time.time()-s)
        
    # except :
    #     # logging.error(traceback.format_exc())
    #     # save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,save_path)
    #     pass

    save_model(epoch,net,optimizer,train_losses,eval_losses,batch_losses,fde_losses)
    if plot:
        plt.plot(train_losses)
        plt.plot(eval_losses)
        plt.show()

        plt.plot(ade_losses)
        plt.plot(fde_losses)
        plt.show()

    return train_losses,eval_losses,batch_losses




def train_sophie(
        generator,
        discriminator, 
        device, 
        train_loader,
        criterion_gan,
        criterion_gen, 
        optimizer_gen,
        optimizer_disc, 
        epoch,
        batch_size,
        obs_length,
        pred_length,
        output_size,
        print_every = 100):
    # model.train()
   
    losses = {
        "mse": 0.,
        "real": 0.,
        "fake": 0.,
        "gen": 0.
    }
    batch_losses = {
        "mse": [],
        "real": [],
        "fake": [],
        "gen": []
    }

    batch_idx = 0
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        inputs,images, labels,ids = data
        inputs,images, labels = inputs.to(device),images.to(device), labels.to(device)

        # train discriminator
        optimizer_disc.zero_grad()
        #### groundtruth batch
        traj_obs = inputs[:,0].view(batch_size,obs_length,output_size)
        traj_pred_real = labels[:,0].view(batch_size,pred_length,output_size)

        real_traj = torch.cat([traj_obs,traj_pred_real], dim = 1)
        real_labels = torch.ones(batch_size).to(device)
        disc_class = discriminator(real_traj).view(batch_size)

        real_loss = criterion_gan(disc_class,real_labels)
        real_loss.backward()

        #### generated batch
        z = generator.gaussian.sample((batch_size,1,)).to(device)
        traj_pred_fake = generator(inputs,images,z)

        fake_traj = torch.cat([traj_obs,traj_pred_fake], dim = 1)
        fake_labels = torch.zeros(batch_size).to(device)
        disc_class = discriminator(fake_traj.detach()).view(batch_size)

        fake_loss = criterion_gan(disc_class,fake_labels)
        fake_loss.backward()
        optimizer_disc.step()

        #################
        # train generator        
        gen_labels = real_labels # we aim for the discriminator to predict 1
       
        optimizer_gen.zero_grad()
        disc_class = discriminator(fake_traj).view(batch_size)
        gen_loss_gan = criterion_gan(disc_class,gen_labels)
        mse_loss = criterion_gen(traj_pred_fake,traj_pred_real)
        loss = gen_loss_gan + mse_loss
        loss.backward()
        optimizer_gen.step()


        losses["mse"] += mse_loss.item()
        losses["real"] += real_loss.item()
        losses["fake"] += fake_loss.item()
        losses["gen"] += gen_loss_gan.item()

        batch_losses["mse"].append(mse_loss.item())
        batch_losses["real"].append(real_loss.item())
        batch_losses["fake"].append(fake_loss.item())
        batch_losses["gen"].append(gen_loss_gan.item())



        if batch_idx % print_every == 0:
            print(batch_idx,time.time()-start_time)   
            print("average mse loss summed over trajectory: {}, gan loss real: {},gan loss fake: {}, gen loss: {}".format(batch_losses["mse"][-1],batch_losses["real"][-1],batch_losses["fake"][-1],batch_losses["gen"][-1]))
    for key in losses:
        losses[key] /= batch_idx    
    print('Epoch n {} Loss: {}, gan loss real: {},gan loss fake: {}, gen loss: {}'.format(epoch,losses["mse"],losses["real"],losses["fake"],losses["gen"]))


    return losses,batch_losses


def eval_sophie(
        generator,
        discriminator, 
        device, 
        eval_loader,
        criterion_gan,
        criterion_gen, 
        epoch,
        batch_size,
        obs_length,
        pred_length,
        output_size,
        scalers_path,
        multiple_scalers,
        print_every = 100):
    # model.train()
   
    # generator.eval()
    # discriminator.eval()
    with torch.no_grad():
        losses = {
            "mse": 0.,
            "real": 0.,
            "fake": 0.,
            "gen": 0.,
            "ade":0.,
            "fde":0.
        }
        # generator.to(device)
        # discriminator.to(device)

        batch_idx = 0
        for batch_idx, data in enumerate(eval_loader):
            inputs,images, labels,ids = data
            inputs,images, labels = inputs.to(device),images.to(device), labels.to(device)

            # train discriminator
            #### groundtruth batch
            traj_obs = inputs[:,0].view(batch_size,obs_length,output_size)
            traj_pred_real = labels[:,0].view(batch_size,pred_length,output_size)

            real_traj = torch.cat([traj_obs,traj_pred_real], dim = 1)
            real_labels = torch.ones(batch_size).to(device)
            disc_class = discriminator(real_traj).view(batch_size)

            real_loss = criterion_gan(disc_class,real_labels)
            
            
            #### generated batch
            z = generator.gaussian.sample((batch_size,1,))
            z = z.to(device)
            # print(generator)
            traj_pred_fake = generator(inputs,images,z)

            fake_traj = torch.cat([traj_obs,traj_pred_fake], dim = 1)
            fake_labels = torch.zeros(batch_size).to(device)
            disc_class = discriminator(fake_traj.detach()).view(batch_size)

            fake_loss = criterion_gan(disc_class,fake_labels)
            
            mse_loss = criterion_gen(traj_pred_fake,traj_pred_real)

            ###################################
            inv_labels,inv_outputs = revert_scaling(ids,traj_pred_real,traj_pred_fake,scalers_path,multiple_scalers)
            inv_outputs = inv_outputs.view(inv_labels.size())

            losses["ade"] += ade_loss(inv_outputs,inv_labels).item()
            losses["fde"] += fde_loss(inv_outputs,inv_labels).item()
            ####################################

            losses["mse"] += mse_loss.item()
            losses["real"] += real_loss.item()
            losses["fake"] += fake_loss.item()



        for key in losses:
            losses[key] /= batch_idx    
        print('Eval Epoch n {} Loss: {}, gan loss real: {},gan loss fake: {}'.format(epoch,losses["mse"],losses["real"],losses["fake"]))
        print('Eval Epoch n {}  ade: {},fde: {}'.format(epoch,losses["ade"],losses["fde"]))
    # generator.train()
    # discriminator.train()
    return losses


def save_sophie(epoch,generator,discriminator,optimizer_gen,optimizer_disc,losses,save_root = "./learning/data/models/" ):

    save_path = save_root + "sophie_{}.tar".format(time.time())


    state = {
        'epoch': epoch,
        'state_dict_d': discriminator.state_dict(),
        'state_dict_g': generator.state_dict(),
        'optimizer_g': optimizer_gen.state_dict(), 
        'optimizer_d': optimizer_disc.state_dict(),  
        'losses': losses
        }
    torch.save(state, save_path)
    
    print("model saved in {}".format(save_path))

def sophie_training_loop(n_epochs,batch_size,generator,discriminator,optimizer_gen,optimizer_disc,device,
        train_loader,eval_loader,obs_length, criterion_gan,criterion_gen, 
        pred_length, output_size,scalers_path,multiple_scalers,plot = True,load_path = None):

    
    losses = {
        "train": {
            "mse": [],
            "real": [],
            "fake": [],
            "gen": []
        },
        "eval": {
            "mse": [],
            "real": [],
            "fake": [],
            "gen": [],
            "ade": [],
            "fde": []

        }        
    }
    
    start_epoch = 0


    if load_path != "":
        print("loading former model from {}".format(load_path))
        checkpoint = torch.load(load_path)
        generator.load_state_dict(checkpoint['state_dict_g'])
        discriminator.load_state_dict(checkpoint['state_dict_d'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_g'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_d'])

        losses = checkpoint["losses"]
        start_epoch = checkpoint["epoch"]


    s = time.time()
    
    # try:
    for epoch in range(start_epoch,n_epochs):
        train_losses,_ = train_sophie(generator,discriminator,device,train_loader,criterion_gan,criterion_gen, 
        optimizer_gen, optimizer_disc,epoch,batch_size,obs_length,pred_length,output_size)

        for key in train_losses:
            losses["train"][key].append(train_losses[key])

        test_losses = eval_sophie(generator,discriminator,device,eval_loader,criterion_gan,criterion_gen,epoch,batch_size,obs_length,pred_length,output_size,scalers_path,multiple_scalers)
        for key in test_losses:
            losses["eval"][key].append(test_losses[key])

        
        print(time.time()-s)
        
    # except :
    #     pass

    save_sophie(epoch,generator,discriminator,optimizer_gen,optimizer_disc,losses)
    if plot:
        plt.plot(losses["train"]["mse"])
        plt.plot(losses["eval"]["mse"])
        plt.show()

    return losses










def revert_scaling(ids,labels,outputs,scalers_root,multiple_scalers = 1):
    if multiple_scalers:
        scaler_ids = ["_".join(id_.split("_")[:-1]) for id_ in ids]
        scalers_path = [scalers_root + id_ +".joblib" for id_ in scaler_ids]
       
        scaler_sample = {}
        for scaler in scalers_path:
            if scaler not in scaler_sample:
                scaler_sample[scaler] = []

                for i,scaler1 in enumerate(scalers_path):
                    if scaler == scaler1:
                        scaler_sample[scaler].append(i)

        for scaler_id in scaler_sample:
            scaler = load(scaler_id)
            samples_ids = scaler_sample[scaler_id]

            sub_labels_torch = labels[samples_ids]
            # b,a,s,i = sub_labels.size()

            sub_labels = sub_labels_torch.contiguous().view(-1,1).cpu().numpy()
            inv_sub_labels = torch.FloatTensor(scaler.inverse_transform(sub_labels)).view(sub_labels_torch.size()).cuda()
            labels[samples_ids] = inv_sub_labels

            sub_outputs_torch = outputs[samples_ids]
            # b,a,s,i = sub_outputs.size()

            sub_outputs = sub_outputs_torch.contiguous().view(-1,1).cpu().detach().numpy()
            inv_sub_outputs = torch.FloatTensor(scaler.inverse_transform(sub_outputs)).view(sub_outputs_torch.size()).cuda()
            outputs[samples_ids] = inv_sub_outputs
        return labels,outputs
    else:

        scaler = load(scalers_root)
        torch_labels = labels.contiguous().view(-1,1).cpu().numpy()
        torch_outputs = outputs.contiguous().view(-1,1).cpu().detach().numpy()

        non_zeros_labels = np.argwhere(torch_labels.reshape(-1))
        non_zeros_outputs = np.argwhere(torch_outputs.reshape(-1))

         
        torch_labels[non_zeros_labels] = np.expand_dims( scaler.inverse_transform(torch_labels[non_zeros_labels].squeeze(-1)) ,axis = 1)
        
        torch_outputs[non_zeros_outputs] = np.expand_dims( scaler.inverse_transform(torch_outputs[non_zeros_outputs].squeeze(-1)),axis = 1)

        inv_labels = torch.FloatTensor(torch_labels).cuda()
        inv_outputs = torch.FloatTensor(torch_outputs).cuda()

        inv_labels = inv_labels.view(labels.size())
        inv_outputs = inv_outputs.view(outputs.size())


        return inv_labels,inv_outputs


def mask_loss(targets):
    b,a = targets.shape[0],targets.shape[1]
    mask = targets.reshape(b,a,-1)
    mask = np.sum(mask,axis = 2)
    mask = mask.reshape(-1)
    mask = np.argwhere(mask).reshape(-1)
    return mask    


def ade_loss(outputs,targets):

    
    outputs = outputs.contiguous().view(-1,2)
    targets = targets.contiguous().view(-1,2)
    mse = nn.MSELoss(reduction= "none")

    mse_loss = mse(outputs,targets )
    mse_loss = torch.sum(mse_loss,dim = 1 )
    mse_loss = torch.sqrt(mse_loss )
    mse_loss = torch.mean(mse_loss )

    return mse_loss

def fde_loss(outputs,targets):

    

    outputs = outputs[:,-1,:]
    targets = targets[:,-1,:]
    mse = nn.MSELoss(reduction= "none")

    mse_loss = mse(outputs,targets )
    mse_loss = torch.sum(mse_loss,dim = 1 )
    mse_loss = torch.sqrt(mse_loss )
    mse_loss = torch.mean(mse_loss )

    return mse_loss