import torch
import torch.nn as nn
# import torch.nn.functional as f
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import time
import os

from classes.datasets import Hdf5Dataset,CustomDataLoader
from classes.rnn_mlp import RNN_MLP
from classes.tcn_mlp import TCN_MLP
from classes.s2s_social_att import S2sSocialAtt
from classes.s2s_spatial_att import S2sSpatialAtt
from classes.social_attention import SocialAttention
from classes.spatial_attention import SpatialAttention
from classes.cnn_mlp import CNN_MLP
import random
import copy


from helpers.training_class import NetTraining

# import helpers.helpers_training as training
# import helpers.net_training as training
import helpers.helpers_training as helpers


import sys
import json
import matplotlib.pyplot as plt
"""
    This script trains a iatcnn model
    The data is loaded using custom dataset
    and pytorch dataloader

    THe model is trained on 80% of the dataset and evaluated
    on the rest.
    The objective function is the mean squared error between
    target sequence and predicted sequence.

    The training evaluates as well the Final Displacement Error

    At the end of the training, model is saved in master/learning/data/models.
    If the training is interrupted, the model is saved.
    The interrupted trianing can be resumed by assigning a file path to the 
    load_path variable.

    Three curves are plotted:
        train ADE
        eval ADE
        eval FDE
"""
# 10332
# 42.7 k


#python learning/rnn_mlp_training.py parameters/data.json parameters/rnn_mlp_training.json parameters/torch_extractors.json parameters/prepare_training.json
def main():
          
    # set pytorch
    torch.manual_seed(10)
    random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(device)
    print(torch.cuda.is_available())
        
    args = sys.argv   

    # data = json.load(open(args[1]))
    # torch_param = json.load(open(args[3]))
    # training_param = json.load(open(args[2]))
    # prepare_param = json.load(open(args[4]))
    training_param = json.load(open("parameters/net_training.json"))
    data = json.load(open("parameters/data.json"))
    torch_param = json.load(open("parameters/torch_extractors.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))


    # open data file and load correct scene lists
    # toy = prepare_param["toy"]
    toy = training_param["toy"]

    data_file = torch_param["split_hdf5"]
    if prepare_param["pedestrian_only"]:
        data_file = torch_param["ped_hdf5"] 

    eval_scenes = prepare_param["eval_scenes"]

    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    if toy:
        print("toy dataset")
        data_file = torch_param["toy_hdf5"]
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
        eval_scenes = test_scenes
        train_eval_scenes = train_scenes

    scenes = [train_eval_scenes,train_scenes,test_scenes,eval_scenes]
    parameters_path = "parameters/networks/{}.json"
    model = training_param["model"]
    
    # select model
    net_params = {}
    net_type = None
    
    if model == "rnn_mlp":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_dim" : net_params["input_dim"],
        "hidden_size" : net_params["hidden_size"],
        "recurrent_layer" : net_params["recurrent_layer"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        "nb_cat": len(prepare_param["types_dic"]),
        "use_types": training_param["use_type"],
        "word_embedding_size": net_params["word_embedding_size"],
        "coord_embedding_size": net_params["coord_embedding_size"],

        "model_name":model,
        "t_pred":training_param["pred_length"],
        "t_obs":training_param["obs_length"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],

        "predict_smooth":prepare_param["smooth"],
        "normalize": training_param["normalize"]
        }
        net_type = RNN_MLP
    elif model == "tcn_mlp":
        net_params = json.load(open(parameters_path.format(model)))
        
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_length" : training_param["obs_length"],
        "output_length" : training_param["pred_length"],
        "num_inputs" : net_params["input_dim"],
        "nb_conv_feat" : net_params["nb_conv_feat"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        # nb_cat: len(prepare_param["types_dic"]),
        "nb_cat": len(prepare_param["types_dic"]),
        "use_types": training_param["use_type"],
        "word_embedding_size": net_params["word_embedding_size"],
        "kernel_size": net_params["kernel_size"],
        "dropout" : net_params["dropout"],

        "coord_embedding_size": net_params["coord_embedding_size"],


        "model_name":model,
        "t_pred":training_param["pred_length"],
        "t_obs":training_param["obs_length"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],

        "predict_smooth":prepare_param["smooth"],
        "normalize": training_param["normalize"]

        }

        net_type= TCN_MLP
    elif model == "cnn_mlp":
        net_params = json.load(open(parameters_path.format(model)))
        
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_length" : training_param["obs_length"],
        "output_length" : training_param["pred_length"],
        "num_inputs" : net_params["input_dim"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        "input_dim" : net_params["input_dim"],

        # nb_cat: len(prepare_param["types_dic"]),
        "nb_cat": len(prepare_param["types_dic"]),
        "kernel_size": net_params["kernel_size"],
        "use_types": training_param["use_type"],
        "coord_embedding_size": net_params["coord_embedding_size"],
        "nb_conv": net_params["nb_conv"],
        "nb_kernel": net_params["nb_kernel"],
        "cnn_feat_size": net_params["cnn_feat_size"],
        "word_embedding_size": net_params["word_embedding_size"],
        
        "model_name":model,
        "t_pred":training_param["pred_length"],
        "t_obs":training_param["obs_length"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],

        "predict_smooth":prepare_param["smooth"],
        "normalize": training_param["normalize"]

        }
        net_type = CNN_MLP

    
    elif model == "s2s_social_attention":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_dim" : net_params["input_dim"],
        "enc_hidden_size" : net_params["enc_hidden_size"],
        "enc_num_layers" : net_params["enc_num_layers"],
        "dec_hidden_size" : net_params["dec_hidden_size"],
        "dec_num_layer" : net_params["dec_num_layer"],

        "embedding_size" : net_params["embedding_size"],
        "output_size" : net_params["output_size"],
        "pred_length" : training_param["pred_length"],
        "projection_layers" : net_params["projection_layers"],
        "enc_feat_embedding" : net_params["enc_feat_embedding"],
        "condition_decoder_on_outputs" : net_params["condition_decoder_on_outputs"],
        
        "model_name":model,
        "t_pred":training_param["pred_length"],
        "t_obs":training_param["obs_length"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],

        "predict_smooth":prepare_param["smooth"],
        "normalize": training_param["normalize"]
        }
        net_type = S2sSocialAtt
    
    elif model == "s2s_spatial_attention":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
        "device" : device,
        "batch_size" : training_param["batch_size"],
        "input_dim" : net_params["input_dim"],
        "enc_hidden_size" : net_params["enc_hidden_size"],
        "enc_num_layers" : net_params["enc_num_layers"],
        "dec_hidden_size" : net_params["dec_hidden_size"],
        "dec_num_layer" : net_params["dec_num_layer"],

        "embedding_size" : net_params["embedding_size"],
        "output_size" : net_params["output_size"],
        "pred_length" : training_param["pred_length"],
        "projection_layers" : net_params["projection_layers"],
        "att_feat_embedding" : net_params["att_feat_embedding"],
        "spatial_projection" : net_params["spatial_projection"],
        "condition_decoder_on_outputs" : net_params["condition_decoder_on_outputs"],
        
        "model_name":model,
        "t_pred":training_param["pred_length"],
        "t_obs":training_param["obs_length"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],

        "predict_smooth":prepare_param["smooth"],
        "normalize": training_param["normalize"]

        }

        net_type = S2sSpatialAtt
    
    elif model == "social_attention":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
            "device" : device,
            "input_dim" : net_params["input_dim"],
            "input_length" : training_param["obs_length"],
            "output_length" : training_param["pred_length"],
            
            # "nb_blocks_transformer" : net_params["nb_blocks"],
            "h" : net_params["h"],
            "dmodel" : net_params["dmodel"],
            # "d_ff_hidden" : 4 * net_params["dmodel"],
            "dk" : int(net_params["dmodel"]/net_params["h"]),
            "dv" : int(net_params["dmodel"]/net_params["h"]),
            "predictor_layers" : net_params["predictor_layers"],
            "pred_dim" : training_param["pred_length"] * net_params["input_dim"] ,
            
            # "convnet_embedding" : net_params["convnet_embedding"],
            "coordinates_embedding_size" : net_params["coordinates_embedding_size"],
            # "convnet_nb_layers" : net_params["convnet_nb_layers"],

            "nb_conv": net_params["nb_conv"],
            "nb_kernel": net_params["nb_kernel"],
            "cnn_feat_size": net_params["cnn_feat_size"],
            "kernel_size" : net_params["kernel_size"],
            
            "dropout_tfr" : net_params["dropout_tfr"],
            "projection_layers":net_params["projection_layers"],
            "use_mha":net_params["use_mha"],
            
            "model_name":model,
            "t_pred":training_param["pred_length"],
            "t_obs":training_param["obs_length"],
            "use_images":net_params["use_images"],
            "use_neighbors":net_params["use_neighbors"],
            "offsets":training_param["offsets"],
            "offsets_input" : training_param["offsets_input"],

            "predict_smooth":prepare_param["smooth"],
            "normalize": training_param["normalize"]

        }
        

        net_type = SocialAttention
    elif model == "spatial_attention":
        net_params = json.load(open(parameters_path.format(model)))
        args_net = {
            "device" : device,
            "input_dim" : net_params["input_dim"],
            "input_length" : training_param["obs_length"],
            "output_length" : training_param["pred_length"],
            "pred_dim" : training_param["pred_length"] * net_params["input_dim"] ,

            
            # "nb_blocks_transformer" : net_params["nb_blocks"],
            "h" : net_params["h"],
            "dmodel" : net_params["dmodel"],
            # "d_ff_hidden" : 4 * net_params["dmodel"],
            "dk" : int(net_params["dmodel"]/net_params["h"]),
            "dv" : int(net_params["dmodel"]/net_params["h"]),
            "predictor_layers" : net_params["predictor_layers"],
            
            # "convnet_embedding" : net_params["convnet_embedding"],
            "coordinates_embedding_size" : net_params["coordinates_embedding_size"],
            # "convnet_nb_layers" : net_params["convnet_nb_layers"],
            "nb_conv": net_params["nb_conv"],
            "nb_kernel": net_params["nb_kernel"],
            "cnn_feat_size": net_params["cnn_feat_size"],
            "kernel_size" : net_params["kernel_size"],


            # "use_tcn" : net_params["use_tcn"],
            # "dropout_tcn" : net_params["dropout_tcn"],
            "dropout_tfr" : net_params["dropout_tfr"],
            "projection_layers":net_params["projection_layers"],
            "spatial_projection":net_params["spatial_projection"],
            "vgg_feature_size":net_params["vgg_feature_size"],


            "use_mha":net_params["use_mha"],
             
            "model_name":model,
            "t_pred":training_param["pred_length"],
            "t_obs":training_param["obs_length"],
            "use_images":net_params["use_images"],
            "use_neighbors":net_params["use_neighbors"],
            "offsets":training_param["offsets"],
            "offsets_input" : training_param["offsets_input"],

            "predict_smooth":prepare_param["smooth"],
            "normalize": training_param["normalize"]

        }
        

        net_type = SpatialAttention

    net = net_type(args_net)

    train_loader,eval_loader,train_dataset,eval_dataset = helpers.load_data_loaders(data,prepare_param,training_param,args_net,data_file,scenes,training_param["batch_size"])
    # train_loader,eval_loader,train_dataset,eval_dataset = helpers.load_data_loaders(data,prepare_param,training_param,net_params,data_file,scenes)
    
    print(net)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr = training_param["lr"])
    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

    args_training = {
        "n_epochs" : training_param["n_epochs"],
        "batch_size" : training_param["batch_size"],
        "device" : device,
        "train_loader" : train_loader,
        "eval_loader" : eval_loader,
        "criterion" : criterion,
        "optimizer" : optimizer,
        "use_neighbors" : net_params["use_neighbors"],
        "scalers_path" : data["scalers"],
        "plot" : training_param["plot"],
        "load_path" : net_params["load_path"],
        "plot_every" : training_param["plot_every"],
        "save_every" : training_param["save_every"],
        "offsets" : training_param["offsets"],
        "offsets_input" : training_param["offsets_input"],
        "unjoint_optimisation": training_param["unjoint_optimisation"],

        "normalized" : training_param["normalize"],
        # "normalized" : prepare_param["normalize"],

        "net" : net,
        "print_every" : training_param["print_every"],
        "nb_grad_plots" : training_param["nb_grad_plots"],
        "nb_sample_plots" : training_param["nb_sample_plots"],
        "train" : training_param["train"],
        "model_name":model,
        "s":training_param["s"],
        "k":training_param["k"],
        "early_stopping_thresh": training_param["early_stopping_thresh"]
         
    }

    if training_param["training_program"] == 1:
        batch_props = training_param["learning_curves_proportions"]
        nb_samples = train_dataset.get_len()
        trainer = NetTraining(args_training)
        trainer.analysis_curves(nb_samples,batch_props,args_training)

    elif training_param["training_program"] == 0:
        trainer = NetTraining(args_training)
        trainer.training_loop()
    elif training_param["training_program"] == 2:
        print("Starting random search")

        report_random_search = data["reports"]+"random_search/random_search.json"
        if os.path.exists(report_random_search):
            os.remove(report_random_search)


        parameters_ = []
        results = []
        for i in range(training_param["nb_run_random_search"]):
            print("-----{}th search".format(i))
            
            args_net_c = copy.deepcopy(args_net)
            hyper_parameters = net_params["hyperparameters"]
            r_hyper = helpers.random_hyperparameters(hyper_parameters)

            while r_hyper["normalize"] == 1 and r_hyper["coordinates_embedding_size"] > 0:
                r_hyper = helpers.random_hyperparameters(hyper_parameters)


            for key in r_hyper:
                args_net_c[key] = r_hyper[key]
            print("-----net parameters")
            print(r_hyper)
            parameters_.append(args_net_c)
            

            net = net_type(args_net_c)
           
            


            train_loader,eval_loader,train_dataset,eval_dataset = helpers.load_data_loaders(data,prepare_param,training_param,args_net_c,data_file,scenes,args_net_c["batch_size"])
    
    
            net = net.to(device)
            optimizer = optim.Adam(net.parameters(),lr = args_net_c["lr"])
            criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

            args_training = {
                "n_epochs" : training_param["n_epochs"],
                "batch_size" : args_net_c["batch_size"],
                "device" : device,
                "train_loader" : train_loader,
                "eval_loader" : eval_loader,
                "criterion" : criterion,
                "optimizer" : optimizer,
                "use_neighbors" : net_params["use_neighbors"],
                "scalers_path" : data["scalers"],
                # "plot" : training_param["plot"],
                "plot" : 0,

                "load_path" : net_params["load_path"],
                "plot_every" : training_param["plot_every"],
                # "save_every" : training_param["save_every"],
                "save_every" : 10000,

                "offsets" : args_net_c["offsets"],
                "offsets_input" : args_net_c["offsets_input"],

                "normalized" : args_net_c["normalize"],
                # "normalized" : prepare_param["normalize"],

                "net" : net,
                "print_every" : training_param["print_every"],
                # "nb_grad_plots" : training_param["nb_grad_plots"],
                "nb_grad_plots" : 0,

                "nb_sample_plots" : 0,
                # "nb_sample_plots" : training_param["nb_sample_plots"],

                "train" : training_param["train"],
                "model_name":model,
                "s":args_net_c["s"],
                "k":training_param["k"],
                "early_stopping_thresh": training_param["early_stopping_thresh"],
                "unjoint_optimisation": training_param["unjoint_optimisation"]
               
            }

            print("-----{}th training".format(i))

            trainer = NetTraining(args_training)
            best_harmonic_fde_ade,best_ade,best_fde = trainer.training_loop()
            results.append(best_harmonic_fde_ade)

            print("harmonic fde ade {}".format(best_harmonic_fde_ade))
            print("-----{}th training done!".format(i))

            # file_ = open(data["reports"]+"random_search.json","rw+")
            if not os.path.exists(report_random_search):
                random_search_dict = {}
            else:  
                random_search_dict = json.load(open(report_random_search,"r"))

            random_search_dict[i] = {"h_ade_fde":best_harmonic_fde_ade,"ade":best_ade,"fde":best_fde,"params":r_hyper}
            print(random_search_dict)
            json.dump(random_search_dict,open(report_random_search,"w+"))

        



if __name__ == "__main__":
    main()


