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
        print_every = 10):
    # model.train()
   

    losses = {
        "mse": 0.,
        "disc": 0.,
        "gen": 0.
    }
    batch_losses = {
        "mse": [],
        "disc": [],
        "gen": []
    }

    batch_idx = 0
    # torch.cuda.synchronize
    start_time = time.time()
    nb_grad_plots = 20
    ids_grads = np.arange(int(train_loader.nb_batches) )
    np.random.shuffle(ids_grads)
    ids_grads = ids_grads[:nb_grad_plots]
    print(ids_grads)


    for batch_idx, data in enumerate(train_loader):

        # torch.cuda.synchronize
        s = time.time()
        inputs, labels,images,ids = data
        inputs,labels,images = inputs.to(device), labels.to(device),images.to(device)

        # torch.cuda.synchronize
        print("data_loading {}".format(time.time()-s))
        s = time.time()

        # train discriminator
        optimizer_disc.zero_grad()
        #### groundtruth batch
        traj_obs = inputs[:,0].view(batch_size,obs_length,output_size) # gt tobs
        traj_pred_real = labels[:,0].view(batch_size,pred_length,output_size) # gt tpred
        real_traj = torch.cat([traj_obs,traj_pred_real], dim = 1) # gt tobs + tpred
        real_labels = torch.ones(batch_size).to(device) # labels == 1

        #### generated batch
        z = generator.gaussian.sample((batch_size,1,)).to(device)
        traj_pred_fake = generator(inputs,images,z) # predicted tpred
        # traj_pred_fake = torch.rand((batch_size,pred_length,output_size)).to(device)
        fake_traj = torch.cat([traj_obs,traj_pred_fake], dim = 1) # predicted tobs + tpred
        fake_labels = torch.zeros(batch_size).to(device) # labels == 0

        traj = torch.cat([real_traj,fake_traj],dim = 0)
        labels = torch.cat([real_labels,fake_labels],dim = 0 )

        ids = torch.randint(2*batch_size,(batch_size,))
        

        disc_class = discriminator(traj[ids].detach()).view(batch_size)
        disc_loss = criterion_gan(disc_class,labels[ids])

        disc_loss.backward()
        if batch_idx in ids_grads:
            helpers.plot_grad_flow(discriminator.named_parameters(),"disc_{}".format(epoch))

        optimizer_disc.step()

        #################
        # train generator    
    
        gen_labels = real_labels # we aim for the discriminator to predict 1
       
        optimizer_gen.zero_grad()
        disc_label = discriminator(fake_traj).view(batch_size)
        gen_loss_gan = criterion_gan(disc_label,gen_labels)
        mse_loss = criterion_gen(traj_pred_fake,traj_pred_real)
        loss = gen_loss_gan + mse_loss
        loss = gen_loss_gan

        loss.backward()

        if batch_idx in ids_grads:
            helpers.plot_grad_flow(generator.named_parameters(),"gen_{}".format(epoch))
        optimizer_gen.step()
   


        losses["mse"] += mse_loss.item()
        losses["disc"] += disc_loss.item()
        losses["gen"] += gen_loss_gan.item()

        batch_losses["mse"].append(mse_loss.item())
        batch_losses["disc"].append(disc_loss.item())
        batch_losses["gen"].append(gen_loss_gan.item())

        # torch.cuda.synchronize
        print("disc gen {}".format(time.time()-s))
        s = time.time()

        if batch_idx % print_every == 0:
            print(batch_idx,time.time()-start_time)   
            # print("average mse loss summed over trajectory: {}, gan loss real: {},gan loss fake: {}, gen loss: {}".format(batch_losses["mse"][-1],batch_losses["real"][-1],batch_losses["fake"][-1],batch_losses["gen"][-1]))
            print("average mse loss summed over trajectory: {}, disc: {}, gen loss: {}".format(batch_losses["mse"][-1],batch_losses["disc"][-1],batch_losses["gen"][-1]))

    for key in losses:
        losses[key] /= (batch_idx    + 1)
    # print('Epoch n {} Loss: {}, gan loss real: {},gan loss fake: {}, gen loss: {}'.format(epoch,losses["mse"],losses["real"],losses["fake"],losses["gen"]))
    print('Epoch n {} Loss: {}, disc loss: {}, gen loss: {}'.format(epoch,losses["mse"],losses["disc"],losses["gen"]))


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
        print_every = 100,
        nb_plots = 16,
        offsets = 0,
        normalized = 0):
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
        nb_batches = eval_loader.nb_batches
        kept_batches_id = np.arange(nb_batches)
        np.random.shuffle(kept_batches_id)

        kept_batches_id = kept_batches_id[:nb_plots]

        kept_samples = []

        batch_idx = 0
        for batch_idx, data in enumerate(eval_loader):
            keep_batch = (batch_idx in kept_batches_id )

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

            if keep_batch:
                kept_sample_id = np.random.randint(0,32)



                kept_samples.append((
                    inputs[kept_sample_id].detach().cpu().numpy(),
                    traj_pred_real[kept_sample_id].unsqueeze(0).detach().cpu().numpy(),
                    traj_pred_fake[kept_sample_id].unsqueeze(0).detach().cpu().numpy()
                    ))

            ###################################
            inv_labels,inv_outputs = traj_pred_real, traj_pred_fake
            if normalized:
                inv_labels,inv_outputs = helpers.revert_scaling(ids,traj_pred_real,traj_pred_fake,scalers_path,multiple_scalers)
                inv_outputs = inv_outputs.view(inv_labels.size())

            losses["ade"] += helpers.ade_loss(inv_outputs,inv_labels).item()
            losses["fde"] += helpers.fde_loss(inv_outputs,inv_labels).item()
            ####################################

            losses["mse"] += mse_loss.item()
            losses["real"] += real_loss.item()
            losses["fake"] += fake_loss.item()
            losses["gen"] += gen_loss.item()
    
        helpers.plot_samples(kept_samples,epoch,n_columns = 1,n_rows = 1)




        for key in losses:
            losses[key] /= (batch_idx    +1)
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
        pred_length, output_size,scalers_path,multiple_scalers,plot = True,load_path = None,plot_every = 5,save_every = 5,
        offsets = 0,normalized = 0):

    
    losses = {
        "train": {
            "mse": [],
            "gen": [],
            "disc": []
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

        print(train_losses)
        for key in train_losses:

            losses["train"][key].append(train_losses[key])

        test_losses = eval_sophie(generator,discriminator,device,eval_loader,criterion_gan,criterion_gen,epoch,batch_size,obs_length,pred_length,output_size,scalers_path,multiple_scalers,offsets,normalized)
        for key in test_losses:
            losses["eval"][key].append(test_losses[key])

        if plot and epoch % plot_every == 0:
            plot_sophie(losses,s,root = "./data/reports/losses/")
        if epoch % save_every == 0:
            save_sophie(epoch,generator,discriminator,optimizer_gen,optimizer_disc,losses)
        print(time.time()-s)
        
    # except Exception as e: 
    #     print(e)

    save_sophie(epoch+1,generator,discriminator,optimizer_gen,optimizer_disc,losses)
    if plot:
        plot_sophie(losses,s,root = "./data/reports/losses/")

        # plt.show()


    return losses

def plot_sophie(losses,idx,root = "./data/reports/samples/"):
    plt.plot(losses["eval"]["ade"],label = "ade")
    plt.plot(losses["eval"]["fde"],label = "fde")
    plt.legend()
    plt.savefig("{}ade_fde_{}.jpg".format(root,idx))
    plt.close()

    # plt.plot(losses["train"]["real"],label = "discriminator_real")
    # plt.plot(losses["train"]["fake"],label = "discriminator_fake")
    plt.plot(losses["train"]["disc"],label = "disc")
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




