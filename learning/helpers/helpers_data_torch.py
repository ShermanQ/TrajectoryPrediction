import csv
import torch
from scipy.spatial.distance import euclidean
import random

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

                # features = np.array([float(f) for f in features])
                # features = features.reshape(nb_objects,t_obs,2)

                # torch.save(features,samples_path+"sample"+sample_id+".pt")
                # torch.save(labels,labels_path+"label"+sample_id+".pt")
                

                
    print(l_ctr,f_ctr,counter,id_)           

def max_object(data_path):
    nb_max = 0
    with open(data_path) as data_csv :
        data_reader = csv.reader(data_csv)
        for data in data_reader:
            nb_objects = int(data[1])
            if nb_objects > nb_max:
                nb_max = nb_objects
    return nb_max


# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     return data, target

# def naive_collate_lstm(batch):
#     data = [item[0][0] for item in batch]
#     target = [item[1][0] for item in batch]
#     return data, target