import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import json 




class Evaluation():
    def __init__(self,data_params,prepare_params):
        data = json.load(open(data_params))
        prepare = json.load(open(prepare_params))

        data_file = data["hdf5_file"]


        # scene lists
        eval_scenes = prepare["eval_scenes"]
        train_eval_scenes = prepare["train_scenes"]
        train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
        test_scenes = prepare["test_scenes"]

        models_path = data["models_evaluation"] + "{}.tar"

        #load lists
        #load model --> à l'appel de la fonction

    def load_model(self,model_name):
        
