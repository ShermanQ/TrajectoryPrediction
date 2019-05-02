
from evaluation.evaluation import Evaluation
from evaluation.classes.tcn_mlp import TCN_MLP
from evaluation.classes.model2a1 import Model2a1

import json
from evaluation.classes.datasets_eval import Hdf5Dataset,CustomDataLoader
import torch
import sys
import learning.helpers.helpers_training as helpers
import torch.nn as nn

# python model_evaluation.py parameters/data.json parameters/prepare_training.json parameters/model_evaluation.json 
def main():
    args = sys.argv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # device = torch.device("cpu")
    print(device)
    print(torch.cuda.is_available())

    eval_params = json.load(open("parameters/model_evaluation.json"))
    data_params = json.load(open("parameters/data.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))

    # report_name = args[4]
    report_name = eval_params["report_name"]

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

    criterions = {
        "loss":helpers.MaskedLoss(nn.MSELoss(reduction="none")),
        "ade":helpers.ade_loss,
        "fde":helpers.fde_loss
    }
    # criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))
    
    # eval_ = Evaluation(args[1],args[2],args[3])
    eval_ = Evaluation("parameters/data.json","parameters/prepare_training.json","parameters/model_evaluation.json")
    torch.manual_seed(21)

    eval_.evaluate(Model2a1,eval_scenes,criterions,device,report_name)

    # eval_.evaluate(Model2a1,["hyang4"],criterions,device,report_name)

    print("done!")


if __name__ == "__main__":
    main()