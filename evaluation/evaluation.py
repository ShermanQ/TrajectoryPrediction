import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import json 
import torch
import sys
import os
from scipy.spatial import distance_matrix
from scipy.stats import norm



from evaluation.classes.datasets_eval import Hdf5Dataset,CustomDataLoader
import learning.helpers.helpers_training as helpers_training
from helpers import get_speeds,get_accelerations

import time




class Evaluation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"
        self.reports_dir = self.data_params["reports_evaluation"] + "{}/"

        self.dynamics = json.load(open(self.data_params["dynamics"]))
        self.delta_t = 1.0/float(self.prepare_params["framerate"])

        self.types_dic = self.prepare_params["types_dic_rev"]

        self.dynamic_threshold = self.eval_params["dynamic_threshold"]




    def load_model(self,model_name,model_class,device):
        print("loading trained model {}".format(model_name))
        checkpoint = torch.load(self.models_path.format(model_name))

        args = checkpoint["args"]

        model = model_class(args)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        return model

    def get_data_loader(self,scene):
        dataset = Hdf5Dataset(
            images_path = self.data_params["prepared_images"],
            hdf5_file= self.data_params["hdf5_file"],
            scene= scene,
            t_obs=self.prepare_params["t_obs"],
            t_pred=self.prepare_params["t_pred"],
            use_images = True,
            data_type = "trajectories",
            # use_neighbors = False,
            use_neighbors = self.eval_params["use_neighbors"],

            use_masks = 1,
            predict_offsets = self.eval_params["offsets"],
            predict_smooth= self.eval_params["predict_smooth"],
            smooth_suffix= self.prepare_params["smooth_suffix"],
            centers = json.load(open(self.data_params["scene_centers"])),
            padding = self.prepare_params["padding"],

            augmentation = 0,
            augmentation_angles = self.eval_params["augmentation_angles"],
            normalize = self.prepare_params["normalize"]
            )
        train_loader = CustomDataLoader( batch_size = self.eval_params["batch_size"],shuffle = False,drop_last = True,dataset = dataset)

        return train_loader
        
        

    # 0: train_eval 1: train 2: eval 3: test
    def evaluate(self,model_class,scenes,criterions,device,report_name,print_every = 500):

        dir_name = self.reports_dir.format(report_name) 
        
        if os.path.exists(dir_name):
            os.system("rm -r {}".format(dir_name))
        os.system("mkdir {}".format(dir_name))

        model = self.load_model(self.eval_params["model_name"],model_class,device)

        nb_criterions = len(criterions)

        losses_scenes = {}
        times = 0
        nb = 0


        for scene in scenes:
            print(scene)
            scene_dict = {}
            losses_dict = {}

            losses_scenes[scene] = {}

            data_loader = self.get_data_loader(scene)
            sample_id = 0
            for data in data_loader:
                

                inputs, labels,types,points_mask, active_mask, imgs = data
                inputs, labels,types, imgs = inputs.to(device), labels.to(device), types.to(device) , imgs.to(device)

                for i,l,t,p,a,img in zip(inputs,labels,types,points_mask,active_mask,imgs):


                    i = i[a]
                    l = l[a]
                    t = t[a]
                    p = p[a]




                    scene_dict[sample_id] = {}
                    losses_dict[sample_id] = {}

                    if sample_id % print_every == 0:
                        print("sample n {}".format(sample_id))

                    
                    a = a.to(device)
                    

                    # print(a)
                    # print(len(a))


                    torch.cuda.synchronize()
                    start = time.time()
                    o = model((i,t,a,p,img))

                    torch.cuda.synchronize()
                    end = time.time() - start

                    times += end 
                    nb += len(a)

                    p = torch.FloatTensor(p).to(device)
                    o = torch.mul(p,o)
                    l = torch.mul(p,l)


                    if self.prepare_params["normalize"]:
                        _,_,i = helpers_training.revert_scaling(l,o,i,self.data_params["scalers"])

                    o = o.view(l.size())
                    i,l,o = helpers_training.offsets_to_trajectories(i.detach().cpu().numpy(),
                                                                        l.detach().cpu().numpy(),
                                                                        o.detach().cpu().numpy(),
                                                                        self.eval_params["offsets"])

                    i,l,o = torch.FloatTensor(i).to(device),torch.FloatTensor(l).to(device),torch.FloatTensor(o).to(device)
                    losses = {}
                    for j,c in enumerate(criterions):
                        criterion = criterions[c]
                        loss = criterion(o, l,p)
                        losses[c] = loss.item()
                        if c not in losses_scenes[scene]:
                            losses_scenes[scene][c] = []
                        losses_scenes[scene][c].append(loss.item())

                    # social loss
                    social_loss,conflict_points = self.conflicts(o.squeeze(1).cpu().numpy())

                    # dynamic_loss = self.dynamic_eval(l.squeeze(1).cpu().numpy(),t.squeeze(1).cpu().numpy())
                    dynamic_loss = self.dynamic_eval(o.squeeze(1).cpu().numpy(),t.squeeze(1).cpu().numpy())


                    if "social" not in losses_scenes[scene]:
                        losses_scenes[scene]["social"] = []
                    losses_scenes[scene]["social"].append(social_loss)
                    losses["social"] = social_loss

                    if "dynamic" not in losses_scenes[scene]:
                        losses_scenes[scene]["dynamic"] = []
                    losses_scenes[scene]["dynamic"].append(dynamic_loss)
                    losses["dynamic"] = dynamic_loss


                    losses_dict[sample_id] = losses


                    

                    scene_dict[sample_id]["inputs"] = i.squeeze(1).cpu().numpy().tolist()
                    scene_dict[sample_id]["labels"] = l.squeeze(1).cpu().numpy().tolist()
                    scene_dict[sample_id]["outputs"] = o.squeeze(1).cpu().numpy().tolist()
                    scene_dict[sample_id]["active_mask"] = a.cpu().numpy().tolist()
                    scene_dict[sample_id]["types"] = t.squeeze(1).cpu().numpy().tolist()
                    scene_dict[sample_id]["points_mask"] = p.squeeze(1).cpu().numpy().tolist()
                    scene_dict[sample_id]["conflict_points"] = conflict_points



                    
                    




                    sample_id += 1

            
            json.dump(scene_dict, open(dir_name + "{}_samples.json".format(scene),"w"),indent= 0)
            json.dump(losses_dict, open(dir_name + "{}_losses.json".format(scene),"w"), indent = 0)

            # print(losses_dict)
                # print(scene_dict)
        
        for scene in losses_scenes:
                for l in losses_scenes[scene]:
                    losses_scenes[scene][l] = np.mean(losses_scenes[scene][l])

        json.dump(losses_scenes, open(dir_name + "losses.json","w"),indent= 0)

        timer = {
            "total_time":times,
            "nb_trajectories":nb,
            "time_per_trajectory":times/nb
        }
        json.dump(timer, open(dir_name + "time.json","w"),indent= 0)


#
#    COnsiders each pair of agent as an interaction
#    Counts the number of problematic interaction
#    averages the percentage over all timesteps
#    returns conflicts coordinates without specifying their timestep
    def conflicts(self,output,threshold = 0.5):
        timesteps = []
        conflict_points = np.array([])
        for t in range(output.shape[1]):
            points = np.array(output[:,t])
            d = distance_matrix(points,points)

            m = (d < 0.5).astype(int) - np.eye(len(points))

            nb_agents_in_conflict = m.sum() / 2.0 # matrix is symmetric
            nb_agents = len(points)**2

            conflict_prop = nb_agents_in_conflict / float(nb_agents) * 100

            timesteps.append(conflict_prop)

            # select points where conflict happens
            ids = np.unique( np.argwhere(m)[:,0] ) 
            if len(ids) > 0:
                points = points[ids]
                if len(conflict_points) > 0:
                    conflict_points = np.concatenate([conflict_points,points], axis = 0)
                else:
                    conflict_points = points



        return np.mean(timesteps),conflict_points.tolist()

# for each agent type, compute accelerations distribution
# over every scenes and trajectories 
# for each type use 5th and 95th percentile as lower and upper bound
# respectively. Every point outside the given range for a given type
# is an outlier.
# for a trajectory of n points, we get m<n accelerations
# we report the proportion of outlier acceleration for a trajectory
# we average on all trajectories
    # def dynamic_eval(self,output,types):
    #     count_per_traj = []
    #     for a in range(output.shape[0]):
    #         coordinates = output[a]
    #         type_ = types[a]
    #         type_ = self.types_dic[str(int(type_))]

    #         accelerations = get_accelerations(get_speeds(coordinates,self.delta_t),self.delta_t)


    #         dynamic_type = self.dynamics[type_]
    #         nb_outliers = 0
    #         for e in accelerations:
    #             if e < dynamic_type["lower_bound"] or e > dynamic_type["upper_bound"]:
    #                 nb_outliers += 1
            
    #         percentage_outlier_points = nb_outliers/len(accelerations) * 100
    #         count_per_traj.append(percentage_outlier_points)

    #     return np.mean(count_per_traj)
        
        
    def dynamic_eval(self,output,types):
        count_per_traj = []
        for a in range(output.shape[0]):
            coordinates = output[a]
            type_ = types[a]
            type_ = self.types_dic[str(int(type_))]

            accelerations = get_accelerations(get_speeds(coordinates,self.delta_t),self.delta_t)


            dynamic_type = self.dynamics[type_]
            # nb_outliers = 0

            props = norm.pdf(accelerations, loc = dynamic_type["mean"], scale=dynamic_type["std"]) 

            nb_outliers = (props < self.dynamic_threshold).astype(int).sum()

            # for e in accelerations:
            #     if e < dynamic_type["lower_bound"] or e > dynamic_type["upper_bound"]:
            #         nb_outliers += 1
            
            percentage_outlier_points = nb_outliers/len(accelerations) * 100
            count_per_traj.append(percentage_outlier_points)

        return np.mean(count_per_traj)



                    

        


        

        




