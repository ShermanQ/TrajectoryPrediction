import csv
import torch
from scipy.spatial.distance import euclidean
import random

"""
INPUT:
    trajectory: sequence of 2D coordinates
    threshold: distance threshold to be traveled during the trajectory the unit is in normalized scene
    (minmax normalization along each axis)
    returns False if the distance traveled during the trajectory
    is lower than threshold, i.e. the agent has not been moving during the trajectory
"""
def is_stopped(trajectory, threshold = 0.01):
    start = trajectory[0]
    end = trajectory[-1]
    d = euclidean(start,end)
    if d < threshold:
        return True
    return False

def extract_tensors(data_path,label_path,samples_path,labels_path):
    counter = 0
    l_ctr =0
    f_ctr = 0
    nb_max = max_object(data_path)
    print(nb_max)
    id_ = 0
    with open(data_path) as data_csv :
        with open(label_path) as label_csv:
            data_reader = csv.reader(data_csv)
            label_reader = csv.reader(label_csv)

            for data,label in zip(data_reader,label_reader):
                sample_id,nb_objects,t_obs,t_pred = data[0],int(data[1]),int(data[2]),int(data[3])
                features = data[4:]
                labels = label[1:]

                features = torch.FloatTensor([float(f) for f in features])
                features = features.view(nb_objects,t_obs,2)
                # features = torch.FloatTensor([float(f) for f in features]+[float(-1) for _ in range( (nb_max-nb_objects) * t_obs * 2)])
                # features = features.view(nb_max,t_obs,2)


                labels = torch.FloatTensor([float(f) for f in labels])
                labels = labels.view(nb_objects,t_pred,2)
                # labels = torch.FloatTensor([float(f) for f in labels] + [float(-1) for _ in range( (nb_max-nb_objects) * t_pred * 2)])

                # labels = labels.view(nb_max,t_pred,2)
                
                # f_stopped = is_stopped(features[0].cpu().detach().numpy())
                l_stopped = is_stopped(labels[0].cpu().detach().numpy())
                # l_stopped = False

                # if f_stopped:
                #     f_ctr += 1
                #     print(features[0].cpu().detach().numpy())
                if l_stopped:
                    l_ctr += 1
                    keep = True if random.random() < 0.0001 else False
                    if keep:
                        torch.save(features,samples_path+"sample"+str(id_)+".pt")
                        torch.save(labels,labels_path+"label"+str(id_)+".pt")
                        id_+= 1

                else:
                    
                    torch.save(features,samples_path+"sample"+str(id_)+".pt")
                    torch.save(labels,labels_path+"label"+str(id_)+".pt")
                    id_+= 1
                    
                counter += 1

"""
"""
def extract_tensors_sophie(data_path,label_path,scene_path,samples_path,labels_path,img_path,images_path,stopped_threshold = 0.01, stopped_prop = 1.0):
    total_samples = 0
    stopped_samples = 0
    stopped_samples_kept = 0

    moving_samples = 0
    nb_max = max_object(data_path)
    print(nb_max)
    id_ = 0
    with open(data_path) as data_csv :
        with open(label_path) as label_csv:
            with open(scene_path) as scene_csv:

                data_reader = csv.reader(data_csv)
                label_reader = csv.reader(label_csv)
                scene_reader = csv.reader(scene_csv)


                for data,label,scene in zip(data_reader,label_reader,scene_reader):
                    sample_id,nb_objects,t_obs,t_pred = data[0],int(data[1]),int(data[2]),int(data[3])
                    features = data[4:]
                    labels = label[1:]

                    features = torch.FloatTensor([float(f) for f in features]+[float(-1) for _ in range( (nb_max-nb_objects) * t_obs * 2)])
                    features = features.view(nb_max,t_obs,2)


                    labels = torch.FloatTensor([float(f) for f in labels] + [float(-1) for _ in range( (nb_max-nb_objects) * t_pred * 2)])
                    labels = labels.view(nb_max,t_pred,2)
                    
                    # is the groundtruth trajectory moving
                    l_stopped = is_stopped(labels[0].cpu().detach().numpy(),stopped_threshold)
                    
                    # if not we keep the sample with probability stopped_prop given by uniform distribution between 0 and 1
                    if l_stopped:
                        stopped_samples += 1
                        keep = True if random.random() < stopped_prop else False
                        if keep:
                            torch.save(features,samples_path+"sample"+str(id_)+".pt")
                            torch.save(labels,labels_path+"label"+str(id_)+".pt")
                            with open(img_path +"img"+str(id_)+".txt","w" ) as img_writer:
                                path_to_img = images_path + scene[0] + ".png"
                                img_writer.write(path_to_img)
                            stopped_samples_kept += 1
                            id_+= 1
                    # if trajectory is movin' add the sample
                    else:
                        
                        torch.save(features,samples_path+"sample"+str(id_)+".pt")
                        torch.save(labels,labels_path+"label"+str(id_)+".pt")
                        with open(img_path +"img"+str(id_)+".txt","w" ) as img_writer:
                            path_to_img = images_path + scene[0] + ".png"
                            img_writer.write(path_to_img)


                        
                        moving_samples += 1
                        id_+= 1
                        
                    total_samples += 1


                
    print("total samples: {}, total moving samples: {}, total stopped samples: {}".format(total_samples,moving_samples,stopped_samples)) 
    print("total samples kept: {}, total stopped samples kept: {}".format(moving_samples + stopped_samples_kept,stopped_samples_kept))           


def max_object(data_path):
    nb_max = 0
    with open(data_path) as data_csv :
        data_reader = csv.reader(data_csv)
        for data in data_reader:
            nb_objects = int(data[1])
            if nb_objects > nb_max:
                nb_max = nb_objects
    return nb_max

