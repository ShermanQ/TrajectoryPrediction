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


import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import torch.nn as nn
import numpy as np
import os 

# python model_evaluation.py parameters/data.json parameters/prepare_training.json parameters/model_evaluation.json 
def main():
    args = sys.argv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
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


    losses_scenes = {}
    times = 0 # sum time for every prediction
    nb = 0 # number of predictions

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
        data_loader = get_data_loader(data_params,data_file,scene,args_net,set_type_test,prepare_param,eval_params)
        
        sample_id = 0
        print_every = 500
        squeeze_dimension = 0 if args_net["use_neighbors"] else 1

        #compute mask for spatial structure
        spatial_masks = scene_mask(scene,images,spatial_annotations,spatial_profiles)
        #get rtio for meter to pixel conversion
        pixel2meters = get_factor(scene,correspondences_trajnet,correspondences_manual)
        
        for batch_idx, data in enumerate(data_loader):
                
                
            # Load data
            inputs, labels,types,points_mask, active_mask, imgs,target_last,input_last = data
            inputs = inputs.to(device)
            labels =  labels.to(device)




            nb_types = len(prepare_param["types_dic"].keys()) 
            # types = torch.FloatTensor(types_ohe(types.cpu().numpy(),nb_types)).to(device)
        
            imgs =  imgs.to(device)        
            active_mask = get_active_mask(points_mask[1])
            points_mask = list(points_mask)

            if not args_net["use_images"]:
                imgs = imgs.repeat(inputs.size()[0],1)

            # penser au cas où target_last et input_last sont des listes vides
            # if not args_net["offsets"]:
            #     target_last = target_last.repeat(inputs.size()[0],1)
            # if not args_net["offsets_input"]:
            #     input_last = input_last.repeat(inputs.size()[0],1)

            # if not args_net["use_images"]:
            #     imgs = imgs.repeat(inputs.size()[0],1)

          



            for i,l,t,p0,p1,a,img,tl,il in zip(inputs,labels,types,points_mask[0],points_mask[1],active_mask,imgs,target_last,input_last):

                    
                    i = i[a]
                    l = l[a]
                    t = t[a]
                    t = t.unsqueeze(squeeze_dimension)

                    #spatial loss
                    spatial_profile_ids = [ types_to_spatial[str(key)] for key in   t.cpu().numpy().astype(int).flatten() ]
                    
                    


                    t = torch.FloatTensor(types_ohe(t.cpu().numpy(),nb_types)).to(device)
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

                    # t = t.unsqueeze(squeeze_dimension)

                    
                    
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
                        

                        loss = criterion(o, l,p)
                        losses[c] = loss.item()
                        # if criterion not in scene of losses report add it
                        if c not in losses_scenes[scene]:
                            losses_scenes[scene][c] = []
                        # append value of criterion in scene/criterion list
                        losses_scenes[scene][c].append(loss.item())


                    # social loss
                    conflict_thresholds = [0.1,0.5,1.0]
                    social_losses = []
                    conflict_points = []

                    # social loss
                    for thresh in conflict_thresholds:
                        ls,pts = conflicts(o.squeeze(squeeze_dimension).cpu().numpy(),thresh)
                        social_losses.append(ls)
                        conflict_points.append(pts)

                        # print("social_".format(thresh))
                        key = "social_" + str(thresh)
                        
                        if key not in losses_scenes[scene]:
                            losses_scenes[scene][key] = []
                        losses_scenes[scene][key].append(ls)
                        losses[key] = ls
                    
                    # dynamic loss
                    speed_len,acc_len = dynamic_eval(
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
                    spatial_losses = []
                    for id_,trajectory_p in zip(spatial_profile_ids,o.squeeze(squeeze_dimension).cpu().numpy()):
                        trajectory_p *= pixel2meters
                        trajectory_p = trajectory_p.astype(np.int32)
                        res = spatial_conflicts(spatial_masks[id_],trajectory_p)
                        spatial_losses.append(res)
                    spatial_loss = np.mean(spatial_losses)

                    if "spatial_loss" not in losses_scenes[scene]:
                        losses_scenes[scene]["spatial_loss"] = []
                    losses_scenes[scene]["spatial_loss"].append(spatial_loss)
                    losses["spatial_loss"] = spatial_loss

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

def types_ohe(types,nb_types):       
    cat = np.arange(1,nb_types+1).reshape(nb_types,1)
    
    ohe = OneHotEncoder(sparse = False,categories = "auto")
    ohe = ohe.fit(cat)

    b,n = types.shape
    # types = types - 1 
    types = ohe.transform(types.reshape(b*n,-1)) 

    types = types.reshape(b,n,nb_types)

    return types

def get_active_mask(mask_target):
    sample_sum = (np.sum(mask_target.reshape(list(mask_target.shape[:2])+[-1]), axis = 2) > 0).astype(int)
                
    active_mask = []
    for b in sample_sum:
        ids = np.argwhere(b.flatten()).flatten()
        active_mask.append(torch.LongTensor(ids))
    return active_mask

def get_data_loader(data_params,data_file,scene,args_net,set_type_test,prepare_param,eval_params):
    dataset = Hdf5Dataset(
            images_path = data_params["prepared_images"],
            hdf5_file= data_file,
            scene_list= [scene], #eval_scenes
            # scene_list= scenes, #eval_scenes

            t_obs=args_net["t_obs"],
            t_pred=args_net["t_pred"],
            set_type = set_type_test, #eval
            use_images = args_net["use_images"],
            data_type = "trajectories",
            # use_neighbors = args_net["use_neighbors"],
            use_neighbors = 1,

            use_masks = 1,
            predict_offsets = args_net["offsets"],
            offsets_input = args_net["offsets_input"],


            predict_smooth= args_net["predict_smooth"],
            smooth_suffix= prepare_param["smooth_suffix"],
            centers = json.load(open(data_params["scene_centers"])),
            padding = prepare_param["padding"],

            augmentation = 0,
            augmentation_angles = [],
            normalize =args_net["normalize"],
            evaluation=1


            )

    data_loader = CustomDataLoader( batch_size = eval_params["batch_size"],shuffle = False,drop_last = False,dataset = dataset,test=0)
    return data_loader

#
#    COnsiders each pair of agent as an interaction
#    Counts the number of problematic interaction
#    averages the percentage over all timesteps
#    returns conflicts coordinates without specifying their timestep
def conflicts(output,threshold = 0.5):
    timesteps = []
    conflict_points = np.array([])
    for t in range(output.shape[1]):
        points = np.array(output[:,t])
        d = distance_matrix(points,points)

        m = (d < threshold).astype(int) - np.eye(len(points))

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

def dynamic_eval(output,types,dynamics,types_dic,delta_t,dynamic_threshold = 0.0):
    count_per_traj_speed = []
    count_per_traj_acc = []

    for a in range(output.shape[0]):
        coordinates = output[a]
        type_ = types[a]
        type_ = types_dic[str(int(type_+1))]

        speeds = get_speeds(coordinates,delta_t)
        accelerations = get_accelerations(speeds,delta_t)
        dynamic_type = dynamics[type_]

        acc_props = norm.pdf(accelerations, loc = dynamic_type["accelerations"]["mean"], scale=dynamic_type["accelerations"]["std"]) 
        speed_props = norm.pdf(accelerations, loc = dynamic_type["speeds"]["mean"], scale=dynamic_type["speeds"]["std"]) 


        nb_outliers_speeds = (speed_props < dynamic_threshold).astype(int).sum()    
        nb_outliers_accs = (acc_props < dynamic_threshold).astype(int).sum()        

        percentage_outlier_points_speed = nb_outliers_speeds/len(speeds) * 100
        percentage_outlier_points_accs = nb_outliers_accs/len(accelerations) * 100

        count_per_traj_speed.append(percentage_outlier_points_speed)
        count_per_traj_acc.append(percentage_outlier_points_accs)

    
    acc_len = np.mean(count_per_traj_acc)
    speed_len = np.mean(count_per_traj_speed)

    return speed_len,acc_len


def get_speed(point1,point2,deltat):
    d = distance.euclidean(point1,point2)
    v = d/deltat
    return v
def get_speeds(coordinates,framerate):
    speeds = []
    for i in range(1,len(coordinates)):
        speed = get_speed(coordinates[i-1],coordinates[i],framerate)
        speeds.append(speed)
    return speeds

def get_acceleration(v1,v2,deltat):
    a = (v2-v1)/deltat
    return a

def get_accelerations(speeds,framerate):
    accelerations = []
    for i in range(1,len(speeds)):
        acceleration = get_acceleration(speeds[i-1],speeds[i],framerate)
        accelerations.append(acceleration)
    return accelerations


# def scene_mask(scene,img_path,class_category,annotations_path):
#         img = mpimg.imread(img_path.format(scene))
#         empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
#         annotations = json.load(open(annotations_path.format(scene)))
#         polygons = []
#         for object_ in annotations["objects"]:
#                 if object_["classTitle"] == class_category:
#                         pts = object_["points"]["exterior"]
#                         a3 = np.array( [pts] ).astype(np.int32)          
#                         cv2.fillPoly( empty_mask, a3, 1 )
#         return empty_mask

# returns array of two masks, first one is pedestrian, second one is wheels
def scene_mask(scene,img_path,annotations_path,spatial_profiles):
        img = mpimg.imread(img_path.format(scene))

        masks = []
        masks_ids = []

        for spatial_profile in spatial_profiles:
        
            empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
            annotations = json.load(open(annotations_path.format(scene)))
            polygons = []
            for object_ in annotations["objects"]:
                    if object_["classTitle"] == spatial_profile:
                            pts = object_["points"]["exterior"]
                            a3 = np.array( [pts] ).astype(np.int32)          
                            cv2.fillPoly( empty_mask, a3, 1 )
            masks.append(empty_mask)
            masks_ids.append(spatial_profiles[spatial_profile])

        arg_ids = np.argsort(masks_ids)
        masks = [masks[i] for i in arg_ids]
        
        return masks

def spatial_conflicts(mask,trajectory_p):
        ctr = 0
        # print(mask.shape)
        for point in trajectory_p:
                #case out of frame
                if point[1] in range(0,mask.shape[0]) and point[0] in range(0,mask.shape[1]):
                    if mask[point[1],point[0]]:
                            ctr += 1
        return ctr / float(len(trajectory_p))

def get_factor(scene,correspondences_trajnet,correspondences_manual):
    if scene in correspondences_trajnet:
        row = correspondences_trajnet[scene]
        pixel2meter_ratio = row["pixel2meter"]
        meter2pixel_ratio = 1/pixel2meter_ratio
    else:
        row = correspondences_manual[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        pixel2meter_ratio = meter_dist/float(pixel_dist)
        meter2pixel_ratio = float(pixel_dist)/meter_dist

    return meter2pixel_ratio




if __name__ == "__main__":
    main()