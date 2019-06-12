from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.evaluation import Evaluation

from classes.rnn_mlp import RNN_MLP
from classes.tcn_mlp import TCN_MLP
from classes.s2s_social_att import S2sSocialAtt
from classes.s2s_spatial_att import S2sSpatialAtt
from classes.social_attention import SocialAttention
from classes.spatial_attention import SpatialAttention
from classes.cnn_mlp import CNN_MLP

from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix,distance
from scipy.stats import norm

from scipy.spatial.distance import euclidean


import matplotlib.image as mpimg
import cv2

import copy

import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import helpers.helpers_evaluation as helpers_evaluation
import torch.nn as nn
import numpy as np
import os 

# python model_evaluation.py parameters/data.json parameters/prepare_training.json parameters/model_evaluation.json 
def main():
    args = sys.argv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    torch.manual_seed(42)
    # device = torch.device("cpu")
    print(device)
    print(torch.cuda.is_available())

    eval_params = json.load(open("parameters/model_evaluation.json"))
    data_params = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))

    prepare_param = json.load(open("parameters/prepare_training.json"))


    # spatial loss variables
    spatial_annotations = data_params["spatial_annotations"]+"{}.jpg.json"
    images = data_params["original_images"]+"{}.jpg"
    types_to_spatial = eval_params["types2spatial"]
    spatial_profiles = eval_params["spatial_profiles"]
    correspondences_trajnet = json.load(open(data_params["pixel2meters_trajnet"]))
    correspondences_manual = json.load(open(data_params["pixel2meters_manual"]))

    #########################

  

    data_file = data_params["hdf5_file"]
    # report_name = args[4]
    report_name = eval_params["report_name"]

    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]


    criterions = {
        "loss":helpers.MaskedLoss(nn.MSELoss(reduction="none")),
        "ade":helpers.ade_loss,
        "fde":helpers.fde_loss
    }

    # model = training_param["model"]
    model_name = eval_params["model_name"]
    models_path = data_params["models_evaluation"] + "{}.tar".format(model_name)

    print("loading trained model {}".format(model_name))
    checkpoint = torch.load(models_path)
    args_net = checkpoint["args"]
    model = args_net["model_name"]  

    net = None
    if model == "rnn_mlp":
        net = RNN_MLP(args_net)
    elif model == "tcn_mlp":   
        net = TCN_MLP(args_net)
    elif model == "cnn_mlp":        
        net = CNN_MLP(args_net)    
    elif model == "s2s_social_attention":        
        net = S2sSocialAtt(args_net)    
    elif model == "s2s_spatial_attention":
        net = S2sSpatialAtt(args_net)    
    elif model == "social_attention":
        net = SocialAttention(args_net)
    elif model == "spatial_attention":     
        net = SpatialAttention(args_net)


    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)
    net.eval()

    print(net)

    scenes = test_scenes
    set_type_test = eval_params["set_type_test"]

    if set_type_test == "train":
        scenes = train_scenes
    elif set_type_test == "eval":
        scenes = eval_scenes
    elif set_type_test == "train_eval":
        scenes = train_eval_scenes

    scenes = ["coupa0"]#########################################################################
    losses_scenes = {}
    times = 0 # sum time for every prediction
    nb = 0 # number of predictions


#############################################################################################
#############################################################################################
    dir_name = data_params["reports_evaluation"] + "{}/".format(report_name)
    sub_dir_name = data_params["reports_evaluation"] + "{}/scene_reports/".format(report_name) 

    
        
    if os.path.exists(dir_name):
        os.system("rm -r {}".format(dir_name))
    os.system("mkdir {}".format(dir_name))
    if os.path.exists(sub_dir_name):
        os.system("rm -r {}".format(sub_dir_name))
    os.system("mkdir {}".format(sub_dir_name))
    for z,scene in enumerate(scenes):
        # if z == 5:
        print(scene)

        scene_dict = {} # save every sample in the scene
        losses_dict = {} # save every sample losses in the scen
        losses_scenes[scene] = {} # init overall report for the scene
        
        # get dataloader
        data_loader = helpers_evaluation.get_data_loader(data_params,data_file,scene,args_net,set_type_test,prepare_param,eval_params)
        
        sample_id = 0
        print_every = 500
        squeeze_dimension = 0 if args_net["use_neighbors"] else 1

        #compute mask for spatial structure
        spatial_masks = helpers_evaluation.scene_mask(scene,images,spatial_annotations,spatial_profiles)
        #get rtio for meter to pixel conversion
        pixel2meters = helpers_evaluation.get_factor(scene,correspondences_trajnet,correspondences_manual)
        
        
        for batch_idx, data in enumerate(data_loader):
                
            if batch_idx  > 1:
                print("break")
                break
            # Load data
            inputs, labels,types,points_mask, active_mask, imgs,target_last,input_last = data
            inputs = inputs.to(device)
            labels =  labels.to(device)




            nb_types = len(prepare_param["types_dic"].keys()) 
        
            imgs =  imgs.to(device)        
            active_mask = helpers_evaluation.get_active_mask(points_mask[1])
            points_mask = list(points_mask)

            if not args_net["use_images"]:
                imgs = imgs.repeat(inputs.size()[0],1)

            for i,l,t,p0,p1,a,img,tl,il in zip(inputs,labels,types,points_mask[0],points_mask[1],active_mask,imgs,target_last,input_last):

                    
                    i = i[a]
                    l = l[a]
                    t = t[a]
                    t = t.unsqueeze(squeeze_dimension)

                    #spatial loss
                    spatial_profile_ids = [ types_to_spatial[str(key)] for key in   t.cpu().numpy().astype(int).flatten() ]
                    
                    


                    t = torch.FloatTensor(helpers_evaluation.types_ohe(t.cpu().numpy(),nb_types)).to(device)
                    if not args_net["use_neighbors"]:
                        t = t.squeeze(squeeze_dimension)



                    p0 = p0[a]
                    p1 = p1[a]
                    tl = tl[a]
                    il = il[a]

                    # cse when there is no neighbors for np arrays
                    if len(a) == 1:
                        p0 = np.expand_dims(p0,axis = 0)
                        p1 = np.expand_dims(p1,axis = 0)
                        tl = np.expand_dims(tl,axis = 0)
                        il = np.expand_dims(il,axis = 0)




                    i = i.unsqueeze(squeeze_dimension)
                    il = np.expand_dims(il,squeeze_dimension)

                    l = l.unsqueeze(squeeze_dimension)
                    tl = np.expand_dims(tl,squeeze_dimension)

                    p0 = np.expand_dims(p0,axis = squeeze_dimension)
                    p1 = np.expand_dims(p1,axis = squeeze_dimension)
                    p = (p0,p1)

                    if sample_id % print_every == 0:
                        print("sample n {}".format(sample_id))
                    
                    a = a.to(device)
                    
                    #predict and count time
                    torch.cuda.synchronize()
                    start = time.time()
                    o = net((i,t,a,p,img))                 
                    torch.cuda.synchronize()
                    end = time.time() - start
                    times += end 
                    nb += len(a)

                    # mask for loss
                    p = torch.FloatTensor(p[1]).to(device)                    
                    o = torch.mul(p,o)
                    l = torch.mul(p,l) # bon endroit?


                    #non testé
                    if args_net["normalize"]:
                        scaler = json.load(open(data_params["scalers"]))

                        # _,_,inputs = helpers.revert_scaling(labels,outputs,inputs,self.scalers_path)            
                        # outputs = outputs.view(labels.size())
                        if args_net["offsets_input"]:
                            meanx =  scaler["standardization"]["meanx"]
                            meany =  scaler["standardization"]["meany"]
                            stdx =  scaler["standardization"]["stdx"]
                            stdy =  scaler["standardization"]["stdy"]
                            i = i.detach().cpu().numpy()

                            i[:,:,:,0] = helpers.revert_standardization(i[:,:,:,0],meanx,stdx)
                            i[:,:,:,1] = helpers.revert_standardization(i[:,:,:,1],meany,stdy)
                            i = torch.FloatTensor(i).to(device)        
                        else:
                            min_ =  scaler["normalization"]["min"]
                            max_ =  scaler["normalization"]["max"]
                            i = helpers.revert_min_max_scale(i.detach().cpu().numpy(),min_,max_)
                            i = torch.FloatTensor(i).to(device)

                   # revert offsets for inputs and outputs
                    o = o.view(l.size())
                    i,l,o = helpers.offsets_to_trajectories(i.detach().cpu().numpy(),
                                                                        l.detach().cpu().numpy(),
                                                                        o.detach().cpu().numpy(),
                                                                        args_net["offsets"],args_net["offsets_input"],tl,il)

                    i,l,o = torch.FloatTensor(i).to(device),torch.FloatTensor(l).to(device),torch.FloatTensor(o).to(device)

                    # compute every standard criterion
                    losses = {}     
                    for j,c in enumerate(criterions):
                        criterion = criterions[c]                        
                        ########################"""
                        # #########################
                        # for those criterion if we want to evaluate we juste pass the parameter to select only first loss"

                        loss = criterion(o.clone(), l.clone(),p.clone(),first_only = 0)
                        loss_unjoint = criterion(o.clone(), l.clone(),p.clone(),first_only = 1)

                        # print(loss,loss_unjoint)
                        losses[c] = loss.item()
                        losses[c+"_unjoint"] = loss_unjoint.item()

                        # if criterion not in scene of losses report add it
                        if c not in losses_scenes[scene]:
                            losses_scenes[scene][c] = []
                        if c+"_unjoint" not in losses_scenes[scene]:
                            losses_scenes[scene][c+"_unjoint"] = []
                        # append value of criterion in scene/criterion list
                        losses_scenes[scene][c].append(loss.item())
                        losses_scenes[scene][c+"_unjoint"].append(loss_unjoint.item())



                    # social loss
                    conflict_thresholds = [0.1,0.5,1.0]
                    social_losses = []
                    conflict_points = []

                    # social loss
                    for thresh in conflict_thresholds:
                        ls,pts = helpers_evaluation.conflicts(o.squeeze(squeeze_dimension).cpu().numpy(),thresh)
                        social_losses.append(ls)
                        conflict_points.append(pts)

                        # print("social_".format(thresh))
                        key = "social_" + str(thresh)
                        
                        if key not in losses_scenes[scene]:
                            losses_scenes[scene][key] = []
                        losses_scenes[scene][key].append(ls)
                        losses[key] = ls
                    
                    # dynamic loss
                    speed_len,acc_len = helpers_evaluation.dynamic_eval(
                        o.squeeze(squeeze_dimension).cpu().numpy(),
                        np.argmax(t.squeeze(squeeze_dimension).cpu().numpy(),-1),
                        json.load(open(data_params["dynamics"])),
                        prepare_param["types_dic_rev"],
                        1.0/float(prepare_param["framerate"]),
                        eval_params["dynamic_threshold"]
                        )

                    if "dynamic_speed" not in losses_scenes[scene]:
                        losses_scenes[scene]["dynamic_speed"] = []
                    losses_scenes[scene]["dynamic_speed"].append(speed_len)
                    losses["dynamic_speed"] = speed_len

                    if "dynamic_acceleration" not in losses_scenes[scene]:
                        losses_scenes[scene]["dynamic_acceleration"] = []
                    losses_scenes[scene]["dynamic_acceleration"].append(acc_len)
                    losses["dynamic_acceleration"] = acc_len

                    # spatial loss
                    # spatial_profile_ids
                    #convert back to pixel
                    spatial_unjoint,spatial_joint = helpers_evaluation.spatial_loss(spatial_profile_ids,spatial_masks,o.squeeze(squeeze_dimension).cpu().numpy(),pixel2meters)
                    
                    # spatial_losses = []
                    # for id_,trajectory_p in zip(spatial_profile_ids,o.squeeze(squeeze_dimension).cpu().numpy()):
                    #     trajectory_p *= pixel2meters
                    #     trajectory_p = trajectory_p.astype(np.int32)
                    #     res = helpers_evaluation.spatial_conflicts(spatial_masks[id_],trajectory_p)
                    #     spatial_losses.append(res)
                    # spatial_loss = np.mean(spatial_losses)

                    if "spatial_joint" not in losses_scenes[scene]:
                        losses_scenes[scene]["spatial_joint"] = []
                    losses_scenes[scene]["spatial_joint"].append(spatial_joint)
                    losses["spatial_joint"] = spatial_joint

                    if "spatial_unjoint" not in losses_scenes[scene]:
                        losses_scenes[scene]["spatial_unjoint"] = []
                    losses_scenes[scene]["spatial_unjoint"].append(spatial_unjoint)
                    losses["spatial_unjoint"] = spatial_unjoint

                    scene_dict[sample_id] = {} # init sample dict in the scene
                    losses_dict[sample_id] = {} # init losses dict for sample in scene
                    # save losses for this sample
                    losses_dict[sample_id] = losses

                    # save sample values
                    t = t.squeeze(squeeze_dimension).cpu().numpy()
                    t = np.argmax(t,-1) + 1
                    scene_dict[sample_id]["inputs"] = i.squeeze(squeeze_dimension).cpu().numpy().tolist()
                    scene_dict[sample_id]["labels"] = l.squeeze(squeeze_dimension).cpu().numpy().tolist()
                    scene_dict[sample_id]["outputs"] = o.squeeze(squeeze_dimension).cpu().numpy().tolist()
                    scene_dict[sample_id]["active_mask"] = a.cpu().numpy().tolist()
                    scene_dict[sample_id]["types"] = t.tolist()
                    scene_dict[sample_id]["points_mask"] = p.squeeze(squeeze_dimension).cpu().numpy().tolist()

                    for thresh,pt in zip(conflict_thresholds,conflict_points):
                        scene_dict[sample_id]["conflict_points_"+str(thresh)] = pt
                

                    sample_id += 1

            # save scene smples and scene losses
            json.dump(scene_dict, open(sub_dir_name + "{}_samples.json".format(scene),"w"),indent= 0)
            json.dump(losses_dict, open(sub_dir_name + "{}_losses.json".format(scene),"w"), indent = 0)

        # for each scene and each criterion, average results   
        for scene in losses_scenes:
            for l in losses_scenes[scene]:
                losses_scenes[scene][l] = np.mean(losses_scenes[scene][l])

        # save mean criterions per scene
        json.dump(losses_scenes, open(dir_name + "losses.json","w"),indent= 0)

        # count the average time per trajectory prediction
        nb = max(1,nb)
        timer = {
            "total_time":times,
            "nb_trajectories":nb,
            "time_per_trajectory":times/nb
        }
        # save the time
        json.dump(timer, open(dir_name + "time.json","w"),indent= 0)   




if __name__ == "__main__":
    main()