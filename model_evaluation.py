
from evaluation.evaluation import Evaluation
from evaluation.classes.tcn_mlp import TCN_MLP
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
        use_neighbors = False,
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

    model_name = "tcn"
    checkpoint = torch.load(data_params["models_evaluation"] + "{}.tar".format(model_name))

    args = checkpoint["args"]
    model_class = TCN_MLP
    model = model_class(args)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

    for data in train_loader:
        inputs, labels, ids,types,points_mask, active_mask, imgs = data
        inputs, labels,types, imgs = inputs.to(device), labels.to(device), types.to(device) , imgs.to(device)
        active_mask = active_mask.to(device)

        for input_,label_,type_,points_mask_,ids_ in zip(inputs,labels,types,points_mask,ids):
            outputs = model((input_,type_,active_mask,points_mask,imgs))

            points_mask_ = torch.FloatTensor(points_mask_).to(device)
            outputs = torch.mul(points_mask_,outputs)
            label_ = torch.mul(points_mask_,label_)


            if prepare_param["normalize"]:
                _,_,input_ = helpers.revert_scaling(ids_,label_,outputs,input_,data_params["scalers"])

            outputs = outputs.view(label_.size())

            input_,label_,outputs = helpers.offsets_to_trajectories(input_.detach().cpu().numpy(),
                                                                label_.detach().cpu().numpy(),
                                                                outputs.detach().cpu().numpy(),
                                                                eval_params["offsets"])

            input_,label_,outputs = torch.FloatTensor(input_).to(device),torch.FloatTensor(label_).to(device),torch.FloatTensor(outputs).to(device)
            

            loss = criterion(outputs, label_,points_mask_)
            print(loss)




        

    # eval_ = Evaluation(args[1],args[2])
    # eval_.load_model("tcn",TCN_MLP)
    print("done!")


if __name__ == "__main__":
    main()