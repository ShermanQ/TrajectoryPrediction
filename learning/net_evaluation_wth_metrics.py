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



  

    data_file = data_params["hdf5_file"]
    # report_name = args[4]
    report_name = eval_params["report_name"]

    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]



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

    s = time.time()
    for z,scene in enumerate(scenes):
        
        print(scene)

        scene_dict = {} # save every sample in the scene
              
        # get dataloader
        data_loader = helpers_evaluation.get_data_loader(data_params,data_file,scene,args_net,set_type_test,prepare_param,eval_params)
        
        sample_id = 0
        #get rtio for meter to pixel conversion

        correspondences_trajnet = json.load(open(data_params["pixel2meters_trajnet"]))
        correspondences_manual = json.load(open(data_params["pixel2meters_manual"]))
        pixel2meters = helpers_evaluation.get_factor(scene,correspondences_trajnet,correspondences_manual)
        
        print(time.time()-s)
        
        for batch_idx, data in enumerate(data_loader):
                
            inputs, labels,types,points_mask, active_mask, imgs,target_last,input_last = data
            inputs = inputs.to(device)
            labels =  labels.to(device)
            imgs =  imgs.to(device)  
            active_mask = active_mask.to(device)
            points_mask = list(points_mask)
            if not args_net["use_images"]:
                imgs = imgs.repeat(inputs.size()[0],1)
            if not args_net["offsets_input"]:
                input_last = np.zeros_like(inputs.cpu().numpy()) 
                

            if not args_net["use_neighbors"]:
                outputs,inputs,types,active_mask,points_mask = helpers_evaluation.predict_naive(inputs,types,active_mask,points_mask,imgs,net,device)


            elif args_net["use_neighbors"] and eval_params["disjoint_evaluation"]:
                outputs,inputs,types,active_mask,points_mask = helpers_evaluation.predict_neighbors_disjoint(inputs,types,active_mask,points_mask,imgs,net,device)

            else:
                outputs = net((inputs,types,active_mask,points_mask,imgs))

        print(time.time()-s)

    print(time.time()-s)
    

if __name__ == "__main__":
    main()