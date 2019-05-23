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

import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import torch.nn as nn
import numpy as np

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

    # toy = prepare_param["toy"]
    # data_file = torch_param["split_hdf5"]
    # if prepare_param["pedestrian_only"]:
    #     data_file = torch_param["ped_hdf5"] 

    data_file = data_params["hdf5_file"]
    # report_name = args[4]
    report_name = eval_params["report_name"]

    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    # load toy scenes
    # toy = prepare_param["toy"]
    # if toy:
    #     print("toy dataset")
    #     train_scenes = prepare_param["toy_train_scenes"]
    #     test_scenes = prepare_param["toy_test_scenes"] 
    #     eval_scenes = test_scenes
    #     train_eval_scenes = train_scenes

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
    times = 0
    nb = 0

    for scene in scenes:
        print(scene)
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

        

        for batch_idx, data in enumerate(data_loader):
                if batch_idx % 1000 == 0:
                    print(batch_idx)
                
                # Load data
                inputs, labels,types,points_mask, active_mask, imgs = data
                inputs = inputs.to(device)
                labels =  labels.to(device)

                nb_types = len(prepare_param["types_dic"].keys()) + 1
                types = torch.FloatTensor(types_ohe(types.cpu().numpy(),nb_types)).to(device)
                # types =  types.to(device)
                

                imgs =  imgs.to(device)        
                active_mask = active_mask.to(device)
                points_mask = list(points_mask)

                if not args_net["use_images"]:
                    imgs = imgs.repeat(inputs.size()[0],1)

                
                sample_sum = (np.sum(points_mask[1].reshape(list(points_mask[1].shape[:2])+[-1]), axis = 2) > 0).astype(int)
                
                active_mask = []
                for b in sample_sum:
                    ids = np.argwhere(b.flatten()).flatten()
                    active_mask.append(torch.LongTensor(ids))





                for i,l,t,p0,p1,a,img in zip(inputs,labels,types,points_mask[0],points_mask[1],active_mask,imgs):

                        i = i[a]
                        l = l[a]
                        t = t[a]
                        p0 = p0[a]
                        p1 = p1[a]

                        print(t)
                        if args_net["use_neighbors"]:
                            i = i.unsqueeze(0)
                            l = l.unsqueeze(0)
                            p0 = np.expand_dims(p0,axis = 0)
                            p1 = np.expand_dims(p1,axis = 0)
                            t = t.unsqueeze(0)
                        else:
                            i = i.unsqueeze(1)
                            l = l.unsqueeze(1)
                            p0 = np.expand_dims(p0,axis = 1)
                            p1 = np.expand_dims(p1,axis = 1)
                            # t = t.unsqueeze(1)
                        p = (p0,p1)

                        # scene_dict[sample_id] = {}
                        # losses_dict[sample_id] = {}

                        # if sample_id % print_every == 0:
                        #     print("sample n {}".format(sample_id))

                        
                        a = a.to(device)
                        

                        # print(a)
                        # print(len(a))


                        torch.cuda.synchronize()
                        start = time.time()
                        o = net((i,t,a,p,img))
                        


                        torch.cuda.synchronize()
                        end = time.time() - start

                        # times += end 
                        # nb += len(a)

                        p = torch.FloatTensor(p[1]).to(device)
                        o = torch.mul(p,o)
                        l = torch.mul(p,l)


                        if args_net["normalize"]:
                            _,_,i = helpers.revert_scaling(l,o,i,data_params["scalers"])

                        o = o.view(l.size())
                        i,l,o = helpers.offsets_to_trajectories(i.detach().cpu().numpy(),
                                                                            l.detach().cpu().numpy(),
                                                                            o.detach().cpu().numpy(),
                                                                            args_net["offsets"])

                        i,l,o = torch.FloatTensor(i).to(device),torch.FloatTensor(l).to(device),torch.FloatTensor(o).to(device)

                        


def types_ohe(types,nb_types):       
    cat = np.arange(nb_types).reshape(nb_types,1)
    print(cat)
    ohe = OneHotEncoder(sparse = False,categories = "auto")
    ohe = ohe.fit(cat)

    b,n = types.shape
    types = ohe.transform(types.reshape(b*n,-1)) 

    types = types.reshape(b,n,nb_types)

    return types


if __name__ == "__main__":
    main()