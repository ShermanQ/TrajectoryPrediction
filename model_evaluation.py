
from evaluation.evaluation import Evaluation
from evaluation.classes.tcn_mlp import TCN_MLP
import json
from evaluation.classes.datasets_eval import Hdf5Dataset,CustomDataLoader

import sys

# python model_evaluation.py parameters/data.json parameters/prepare_training.json
def main():
    args = sys.argv

    eval_params = json.load(open("parameters/model_evaluation.json"))
    data = json.load(open("parameters/data.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))


    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    # load toy scenes
    if toy:
        print("toy dataset")
        train_scenes = prepare_param["toy_train_scenes"]
        test_scenes = prepare_param["toy_test_scenes"] 
        eval_scenes = test_scenes
        train_eval_scenes = train_scenes

    # load hdf5 file orgnized per scenes
    data_file = data["hdf5_file"]

    scene = train_eval_scenes[0]

    # create dataset
    dataset = Hdf5Dataset(
        images_path = data["prepared_images"],
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
        centers = json.load(open(data["scene_centers"])),
        padding = prepare_param["padding"],

        augmentation = 0,
        augmentation_angles = eval_params["augmentation_angles"],
        normalize =prepare_param["normalize"]
        )

    #initialize dataloader
    train_loader = CustomDataLoader( batch_size = eval_params["batch_size"],shuffle = False,drop_last = True,dataset = dataset)

    for data in train_loader:
        print(1)

    # eval_ = Evaluation(args[1],args[2])
    # eval_.load_model("tcn",TCN_MLP)
    print("done!")


if __name__ == "__main__":
    main()