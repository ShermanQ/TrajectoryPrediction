
from evaluation.evaluation import Evaluation
from evaluation.classes.tcn_mlp import TCN_MLP
from evaluation.classes.model2a1 import Model2a1

import json
from evaluation.classes.datasets_eval import Hdf5Dataset,CustomDataLoader
import torch
import sys
import learning.helpers.helpers_training as helpers
import torch.nn as nn

# python model_evaluation.py parameters/data.json parameters/prepare_training.json
def main():
    args = sys.argv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # device = torch.device("cpu")
    print(device)
    print(torch.cuda.is_available())

    eval_params = json.load(open("parameters/model_evaluation.json"))
    data_params = json.load(open("parameters/data.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))


    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    # load toy scenes
    toy = prepare_param["toy"]
    if toy:
        print("toy dataset")
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
        eval_scenes = test_scenes
        train_eval_scenes = train_scenes

    # load hdf5 file orgnized per scenes
    data_file = data_params["hdf5_file"]

    scene = train_eval_scenes[0]

    # create dataset
    dataset = Hdf5Dataset(
        images_path = data_params["prepared_images"],
        hdf5_file= data_file,
        scene= scene,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        use_images = True,
        data_type = "trajectories",
        # use_neighbors = False,
        use_neighbors = eval_params["use_neighbors"],

        use_masks = 1,
        predict_offsets = eval_params["offsets"],
        predict_smooth= eval_params["predict_smooth"],
        smooth_suffix= prepare_param["smooth_suffix"],
        centers = json.load(open(data_params["scene_centers"])),
        padding = prepare_param["padding"],

        augmentation = 0,
        augmentation_angles = eval_params["augmentation_angles"],
        normalize =prepare_param["normalize"]
        )

    #initialize dataloader
    train_loader = CustomDataLoader( batch_size = eval_params["batch_size"],shuffle = False,drop_last = True,dataset = dataset)

    # model_name = "tcn"
    model_name = eval_params["model_name"]

    checkpoint = torch.load(data_params["models_evaluation"] + "{}.tar".format(model_name))

    args = checkpoint["args"]
    model_class = TCN_MLP
    # model_class = Model2a1

    model = model_class(args)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

    for data in train_loader:
        inputs, labels,types,points_mask, active_mask, imgs = data
        inputs, labels,types, imgs = inputs.to(device), labels.to(device), types.to(device) , imgs.to(device)

        for i,l,t,p,a,img in zip(inputs,labels,types,points_mask,active_mask,imgs):
            a = a.to(device)

            o = model((i,t,a,p,img))

            p = torch.FloatTensor(p).to(device)
            o = torch.mul(p,o)
            l = torch.mul(p,l)


            if prepare_param["normalize"]:
                _,_,i = helpers.revert_scaling(l,o,i,data_params["scalers"])

            o = o.view(l.size())

            i,l,o = helpers.offsets_to_trajectories(i.detach().cpu().numpy(),
                                                                l.detach().cpu().numpy(),
                                                                o.detach().cpu().numpy(),
                                                                eval_params["offsets"])

            i,l,o = torch.FloatTensor(i).to(device),torch.FloatTensor(l).to(device),torch.FloatTensor(o).to(device)
            

            loss = criterion(o, l,p)
            print(loss)




        

    # eval_ = Evaluation(args[1],args[2])
    # eval_.load_model("tcn",TCN_MLP)
    print("done!")


if __name__ == "__main__":
    main()