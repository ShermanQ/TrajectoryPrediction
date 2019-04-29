import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import json 
import torch
import sys

from evaluation.classes.datasets_eval import Hdf5Dataset,CustomDataLoader
import learning.helpers.helpers_training as helpers





class Evaluation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"


    def load_model(self,model_name,model_class,device):
        print("loading trained model {}".format(model_name))
        checkpoint = torch.load(self.models_path.format(model_name))

        args = checkpoint["args"]

        model = model_class(args)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        return model

    def get_data_loader(self,scene):
        dataset = Hdf5Dataset(
            images_path = self.data_params["prepared_images"],
            hdf5_file= self.data_params["hdf5_file"],
            scene= scene,
            t_obs=self.prepare_params["t_obs"],
            t_pred=self.prepare_params["t_pred"],
            use_images = True,
            data_type = "trajectories",
            # use_neighbors = False,
            use_neighbors = self.eval_params["use_neighbors"],

            use_masks = 1,
            predict_offsets = self.eval_params["offsets"],
            predict_smooth= self.eval_params["predict_smooth"],
            smooth_suffix= self.prepare_params["smooth_suffix"],
            centers = json.load(open(self.data_params["scene_centers"])),
            padding = self.prepare_params["padding"],

            augmentation = 0,
            augmentation_angles = self.eval_params["augmentation_angles"],
            normalize = self.prepare_params["normalize"]
            )
        train_loader = CustomDataLoader( batch_size = self.eval_params["batch_size"],shuffle = False,drop_last = True,dataset = dataset)

        return train_loader
        
        

    # 0: train_eval 1: train 2: eval 3: test
    def evaluate(self,model_class,scenes,criterions,device,print_every = 500):
        model = self.load_model(self.eval_params["model_name"],model_class,device)
        
        for scene in scenes:
            print(scene)
            scene_dict = {}
            losses_dict = {}

            data_loader = self.get_data_loader(scene)
            sample_id = 0
            for data in data_loader:
                

                inputs, labels,types,points_mask, active_mask, imgs = data
                inputs, labels,types, imgs = inputs.to(device), labels.to(device), types.to(device) , imgs.to(device)

                for i,l,t,p,a,img in zip(inputs,labels,types,points_mask,active_mask,imgs):
                    scene_dict[sample_id] = {}
                    losses_dict[sample_id] = {}

                    if sample_id % print_every == 0:
                        print("sample n {}".format(sample_id))
                    a = a.to(device)
                    o = model((i,t,a,p,img))

                    p = torch.FloatTensor(p).to(device)
                    o = torch.mul(p,o)
                    l = torch.mul(p,l)


                    if self.prepare_params["normalize"]:
                        _,_,i = helpers.revert_scaling(l,o,i,self.data_params["scalers"])

                    o = o.view(l.size())
                    i,l,o = helpers.offsets_to_trajectories(i.detach().cpu().numpy(),
                                                                        l.detach().cpu().numpy(),
                                                                        o.detach().cpu().numpy(),
                                                                        self.eval_params["offsets"])

                    i,l,o = torch.FloatTensor(i).to(device),torch.FloatTensor(l).to(device),torch.FloatTensor(o).to(device)
                    losses = {}
                    for c in criterions:
                        criterion = criterions[c]
                        loss = criterion(o, l,p)
                        losses[c] = loss.item()
                    

                    losses_dict[sample_id] = losses
                    

                    scene_dict[sample_id]["inputs"] = i.cpu().numpy()
                    scene_dict[sample_id]["labels"] = l.cpu().numpy()
                    scene_dict[sample_id]["outputs"] = o.cpu().numpy()
                    scene_dict[sample_id]["active_mask"] = a.cpu().numpy()
                    scene_dict[sample_id]["types"] = t.cpu().numpy()
                    scene_dict[sample_id]["points_mask"] = p.cpu().numpy()


                    




                    sample_id += 1
            print(losses_dict)
                # print(scene_dict)

                    

        


        

        




