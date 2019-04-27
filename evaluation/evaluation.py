import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import json 
import torch




class Evaluation():
    def __init__(self,data_params,prepare_params):
        data = json.load(open(data_params))
        prepare = json.load(open(prepare_params))

        data_file = data["hdf5_file"]


        # scene lists
        self.eval_scenes = prepare["eval_scenes"]
        self.train_eval_scenes = prepare["train_scenes"]
        self.train_scenes = [scene for scene in self.train_eval_scenes if scene not in self.eval_scenes]
        self.test_scenes = prepare["test_scenes"]

        self.lists = {
            0:self.train_eval_scenes,
            1:self.train_scenes,
            2:self.eval_scenes,
            3:self.test_scenes
        }

        self.models_path = data["models_evaluation"] + "{}.tar"

        #load lists
        #load model --> à l'appel de la fonction

    def load_model(self,model_name,model_class):
        print("loading trained model {}".format(model_name))
        checkpoint = torch.load(self.models_path.format(model_name))

        args = checkpoint["args"]

        model = model_class(args)
        model.load_state_dict(checkpoint['state_dict'])

        return model

    # 0: train_eval 1: train 2: eval 3: test
    def evaluate(self,model_name,model_class,train_scenes,test_scenes):
        model = self.load_model(model_name,model_class)
        model.eval()

        train_scenes = self.lists[train_scenes]
        test_scenes = self.lists[test_scenes]

        




