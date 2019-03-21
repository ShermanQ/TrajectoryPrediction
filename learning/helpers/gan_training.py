import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import helpers.helpers_training as helpers
import os 

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
        inputs, labels,images,ids = data
        inputs,labels,images = inputs.to(device), labels.to(device),images.to(device)

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
            inputs, labels,images,ids = data
            inputs,labels,images = inputs.to(device), labels.to(device),images.to(device)

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
            gen_loss = criterion_gan(disc_class,real_labels)

            
            mse_loss = criterion_gen(traj_pred_fake,traj_pred_real)

            ###################################
            inv_labels,inv_outputs = helpers.revert_scaling(ids,traj_pred_real,traj_pred_fake,scalers_path,multiple_scalers)
            inv_outputs = inv_outputs.view(inv_labels.size())

            losses["ade"] += helpers.ade_loss(inv_outputs,inv_labels).item()
            losses["fde"] += helpers.fde_loss(inv_outputs,inv_labels).item()
            ####################################

            losses["mse"] += mse_loss.item()
            losses["real"] += real_loss.item()
            losses["fake"] += fake_loss.item()
            losses["gen"] += gen_loss.item()




        for key in losses:
            losses[key] /= batch_idx    
        print('Eval Epoch n {} Loss: {}, gan loss real: {},gan loss fake: {}'.format(epoch,losses["gen"],losses["real"],losses["fake"]))
        print('Eval Epoch n {}  ade: {},fde: {}'.format(epoch,losses["ade"],losses["fde"]))
    # generator.train()
    # discriminator.train()
    return losses


def save_sophie(epoch,generator,discriminator,optimizer_gen,optimizer_disc,losses,save_root = "./learning/data/models/" ):

    # save_path = save_root + "sophie_{}.tar".format(time.time())

    dirs = os.listdir(save_root)

    save_path = save_root + "sophie_{}_{}.tar".format(epoch,time.time())


    state = {
        'epoch': epoch,
        'state_dict_d': discriminator.state_dict(),
        'state_dict_g': generator.state_dict(),
        'optimizer_g': optimizer_gen.state_dict(), 
        'optimizer_d': optimizer_disc.state_dict(),  
        'losses': losses
        }


    # state = {
    #     'state_dict_d': discriminator.state_dict(),
    #     'state_dict_g': generator.state_dict(), 
        
    #     }
    torch.save(state, save_path)

    for dir_ in dirs:
        os.remove(save_root+dir_)
    
    print("model saved in {}".format(save_path))

def sophie_training_loop(n_epochs,batch_size,generator,discriminator,optimizer_gen,optimizer_disc,device,
        train_loader,eval_loader,obs_length, criterion_gan,criterion_gen, 
        pred_length, output_size,scalers_path,multiple_scalers,plot = True,load_path = None,plot_every = 5,save_every = 5):

    
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
    
    try:
        for epoch in range(start_epoch,n_epochs):
            train_losses,_ = train_sophie(generator,discriminator,device,train_loader,criterion_gan,criterion_gen, 
            optimizer_gen, optimizer_disc,epoch,batch_size,obs_length,pred_length,output_size)

            print(train_losses)
            for key in train_losses:

                losses["train"][key].append(train_losses[key])

            test_losses = eval_sophie(generator,discriminator,device,eval_loader,criterion_gan,criterion_gen,epoch,batch_size,obs_length,pred_length,output_size,scalers_path,multiple_scalers)
            for key in test_losses:
                losses["eval"][key].append(test_losses[key])

            if plot and epoch % plot_every == 0:
                plot_sophie(losses,s,root = "./data/reports/")
            if epoch % save_every == 0:
                save_sophie(epoch,generator,discriminator,optimizer_gen,optimizer_disc,losses)
            print(time.time()-s)
        
    except :
        pass

    save_sophie(epoch+1,generator,discriminator,optimizer_gen,optimizer_disc,losses)
    if plot:
        plot_sophie(losses,s,root = "./data/reports/")

        # plt.show()


    return losses

def plot_sophie(losses,idx,root = "./data/reports/"):
    plt.plot(losses["eval"]["ade"],label = "ade")
    plt.plot(losses["eval"]["fde"],label = "fde")
    plt.legend()
    plt.savefig("{}ade_fde_{}.jpg".format(root,idx))
    plt.close()

    plt.plot(losses["train"]["real"],label = "discriminator_real")
    plt.plot(losses["train"]["fake"],label = "discriminator_fake")
    plt.plot(losses["train"]["gen"],label = "generator")
    plt.legend()


    plt.savefig("{}train_gan_{}.jpg".format(root,idx))
    plt.close()

    plt.plot(losses["eval"]["real"],label = "discriminator_real")
    plt.plot(losses["eval"]["fake"],label = "discriminator_fake")
    plt.plot(losses["eval"]["gen"],label = "generator")
    plt.legend()


    plt.savefig("{}eval_gan_{}.jpg".format(root,idx))
    plt.close()

    plt.plot(losses["train"]["mse"],label = "train_mse")
    plt.plot(losses["eval"]["mse"],label = "eval_mse")
    plt.legend()

    plt.savefig("{}mse_{}.jpg".format(root,idx))
    plt.close()




