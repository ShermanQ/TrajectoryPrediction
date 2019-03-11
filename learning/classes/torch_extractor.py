import csv
import torch
from scipy.spatial.distance import euclidean
import random
import shutil
import os
import json


class TorchExtractor():
    def __init__(self,data,torch_params):
        data = json.load(open(data))
        torch_params = json.load(open(torch_params))

        self.prepared_samples = data["prepared_samples_grouped"]
        self.prepared_labels = data["prepared_labels_grouped"]
        self.ids_path = data["prepared_ids"]
        self.samples_ids = json.load(open(self.ids_path))["ids"]
        self.kept_ids = torch_params["ids_path"]

        self.prepared_images = data["prepared_images"]
        self.stopped_threshold = torch_params["stopped_threshold"]
        self.stopped_prop = torch_params["stopped_prop"]

        self.samples_torch = torch_params["samples_torch"]
        self.labels_torch = torch_params["labels_torch"]
        self.img_links_torch = torch_params["img_links_torch"]
        self.new_padding = torch_params["new_padding"]
        self.old_padding = torch_params["old_padding"]


        self.input_size = 2

    def extract_tensors_sophie(self):
        shutil.rmtree(self.samples_torch)
        shutil.rmtree(self.labels_torch)
        shutil.rmtree(self.img_links_torch)

        try:  
            os.mkdir(self.samples_torch)
            os.mkdir(self.labels_torch)
            os.mkdir(self.img_links_torch)

        except OSError:  
            print ("Creation of one of the directories failed")
        
        total_samples = 0
        stopped_samples = 0
        stopped_samples_kept = 0

        moving_samples = 0
        nb_max = self.__max_object()
        print(nb_max)
        id_ = 0
        print(self.prepared_samples)

        kept_ids = []
        with open(self.prepared_samples) as data_csv :
            with open(self.prepared_labels) as label_csv:
                data_reader = csv.reader(data_csv)
                label_reader = csv.reader(label_csv)


                for data,label,sample_id in zip(data_reader,label_reader,self.samples_ids):
                    nb_objects,t_obs,t_pred = int(data[1]),int(data[2]),int(data[3])

                    features = data[4:]
                    labels = label[1:]

                    features = torch.FloatTensor([float(f) if float(f) != self.old_padding else self.new_padding for f in features  ]+[float(self.new_padding) for _ in range( (nb_max-nb_objects) * t_obs * self.input_size)])
                    features = features.view(nb_max,t_obs,self.input_size)
                    
                    # features = torch.FloatTensor([float(f) for f in features]+[float(-1) for _ in range( (nb_max-nb_objects) * t_obs * self.input_size)])
                    # features = features.view(nb_max,t_obs,self.input_size)


                    labels = torch.FloatTensor([float(f) if float(f) != self.old_padding else self.new_padding for f in labels] + [float(self.new_padding) for _ in range( (nb_max-nb_objects) * t_pred * self.input_size)])
                    labels = labels.view(nb_max,t_pred,self.input_size)
                    
                    # is the groundtruth trajectory moving
                    l_stopped = self.__is_stopped(labels[0].cpu().detach().numpy())
                    
                    # if not we keep the sample with probability stopped_prop given by uniform distribution between 0 and 1
                    if l_stopped:
                        stopped_samples += 1
                        keep = True if random.random() < self.stopped_prop else False
                        if keep:
                            kept_ids.append(sample_id)


                            torch.save(features,self.samples_torch+"sample_"+sample_id+".pt")
                            torch.save(labels,self.labels_torch+"label_"+sample_id+".pt")

                            with open(self.img_links_torch +"img_"+sample_id+".txt","w" ) as img_writer:

                                sample_scene = "_".join(sample_id.split("_")[:-1])
                                path_to_img = self.prepared_images + sample_scene + ".jpg"
                                img_writer.write(path_to_img)
                            stopped_samples_kept += 1
                            id_+= 1
                    # if trajectory is movin' add the sample
                    else:
                        kept_ids.append(sample_id)
                        torch.save(features,self.samples_torch+"sample_"+sample_id+".pt")
                        torch.save(labels,self.labels_torch+"label_"+sample_id+".pt")
                        with open(self.img_links_torch +"img_"+sample_id+".txt","w" ) as img_writer:

                            sample_scene = "_".join(sample_id.split("_")[:-1])
                            path_to_img = self.prepared_images + sample_scene + ".jpg"
                            img_writer.write(path_to_img)                        
                        moving_samples += 1
                        id_+= 1
                        
                    total_samples += 1

        ids_json = json.load(open(self.ids_path))
        ids_json["ids"] = kept_ids
        json.dump(ids_json,open(self.kept_ids,"w"))


                    
        print("total samples: {}, total moving samples: {}, total stopped samples: {}".format(total_samples,moving_samples,stopped_samples)) 
        print("total samples kept: {}, total stopped samples kept: {}".format(moving_samples + stopped_samples_kept,stopped_samples_kept))           

    """
    INPUT:
        trajectory: sequence of 2D coordinates
        threshold: distance threshold to be traveled during the trajectory the unit is in normalized scene
        (minmax normalization along each axis)
        returns False if the distance traveled during the trajectory
        is lower than threshold, i.e. the agent has not been moving during the trajectory
    """
    def __is_stopped(self,trajectory ):
        start = trajectory[0]
        end = trajectory[-1]
        d = euclidean(start,end)
        if d < self.stopped_threshold:
            return True
        return False

    """
    """


    def __max_object(self):
        nb_max = 0
        with open(self.prepared_samples) as data_csv :
            data_reader = csv.reader(data_csv)
            for data in data_reader:
                nb_objects = int(data[1])
                if nb_objects > nb_max:
                    nb_max = nb_objects
        return nb_max

