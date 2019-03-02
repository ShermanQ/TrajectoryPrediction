import csv
from itertools import islice
import extractors.helpers as helpers 
import json
import numpy as np
import time
from itertools import tee
import os

  #
def round_coordinates(coordinates,decimal_nb = 2):
    x = coordinates[0]
    y = coordinates[1]
    x = int( x * 10**decimal_nb)/float(10**decimal_nb)
    y = int( y * 10**decimal_nb)/float(10**decimal_nb)
    return [x,y]

"""
    in: 
        start:frame number where the trajectory begins
        stop: frame number where the trajectory stops
        frames_path: path to the file containing frames
    out:
        returns ids but the lists are filled with the coordinates
        of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
"""
def get_neighbors(frames):
# def get_neighbors_coordinates(start,stop,frames,ids):
    ids = {}
    # with open(frames_path) as frames:
    for i,frame in enumerate(frames):
        frame = json.loads(frame)

        for id_ in frame:
            if id_ != "frame":
                if int(id_) not in ids:
                    ids[int(id_)] = [[-1,-1] for j in range(i)]
        
        for id_ in ids:
            if str(id_) in frame:
                ids[id_].append(round_coordinates(frame[str(id_)]["coordinates"]))
            else:
                ids[id_].append([-1,-1])
    return ids


"""
    in:
        t_obs: number of observed frames
        ids: ids  filled with the coordinates
        id_: id of the neighbor to test
        of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        current_frame: the frame of the trajectory to be considered
    out: 
        True if the neighboor appears during observation time
"""
# def add_neighbor(t_obs,ids,id_,current_frame):
#     add = False
#     for p in ids[id_][current_frame:current_frame+t_obs]:
#         if p != [-1,-1]:
#             return True
#     return add

"""
    in:
        t_obs: number of observed frames
        ids: ids  filled with the coordinates
        id_: id of the neighbor to test
        of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        current_frame: the frame of the trajectory to be considered
    out: 
        True if the neighboor appears during the last time step of the observation time
"""
def add_neighbor(t_obs,ids,id_,current_frame):
    if ids[id_][current_frame:current_frame+t_obs][-1] != [-1,-1]:
        return True
    return False

"""
    in:
        ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        id_: id of the neighbor to test
        
        t_obs: number of observed frames
        t_pred: number of frames to predict
        current_frame: the frame of the trajectory to be considered
    out: 
        feature: for the given id, steps between current_frame and current_frame + t_obs 2D COORDINATES ARE FLATTENED
        label: for the given id, steps between current_frame+t_obs and current_frame+t_obs+t_pred

"""
def feature_label(ids,id_,t_obs,t_pred,current_frame,padding = -1):

    feature = ids[id_][current_frame:current_frame+t_obs]
    ctr = 0
    for p in feature:
        if p[0] == padding:
            ctr += 1
    feature = np.array(feature[ctr:] + feature[:ctr])
    feature = feature.flatten().tolist()
    label = np.array(ids[id_][current_frame+t_obs:current_frame+t_obs+t_pred]).flatten().tolist()
    return feature,label

"""
    in:
        len_traj: length of the main trajectory
        shift: the size of the step between to feature extraction for the main trajectory
        t_obs: number of observed frames
        t_pred: number of frames to predict
        current_id: id of the main trajectory
        ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        current_frame: the frame of the trajectory to be considered
    out:
        features: first to appear is the main trajectory, then the neighboors that appears during observation
                  everything is flattened: let xij, i traj_id, j time_step [x00,y00,...x0n,y0n, ... , xn0,yn0,...xnn,ynn,]
                  j < current_frame + t_obs
        labels: same idea but for prediction time


"""
def features_labels(len_traj,shift,t_obs,t_pred,current_id,ids,current_frame):
    
    features = []
    labels = []
    if current_frame + t_obs + t_pred -1 < len_traj:
        

        feature,label = feature_label(ids,current_id,t_obs,t_pred,current_frame)

        # if feature[0] == -1:
        #     print("fdggsdg")

        features.append(feature)
        labels.append(label)

        for id_ in sorted(ids):
            if id_ != current_id:
                
                if add_neighbor(t_obs,ids,id_,current_frame):


                    feature,label = feature_label(ids,id_,t_obs,t_pred,current_frame)

                    features.append(feature)
                    labels.append(label)

        features = np.array(features).flatten().tolist()
        labels = np.array(labels).flatten().tolist()
    return features,labels


"""
    in:
        ids: ids  filled with the coordinates of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
        features: see features_labels
        labels: see features_labels
        data_writer: csv writer for file containing features
        label_writer: csv writer for file containing labels
        sample_id: running id of the number of training sample
    out:
"""
def persist_data(ids,parameters,current_id,current_frame,features,labels,data_writer,label_writer,scene_writer,sample_id,scene):
    

    # ids_list = sorted([id_ for id_ in ids if add_neighbor(parameters["t_obs"],ids,id_,current_frame)])
    # nb_objects = len(ids_list)
    nb_objects = int((len(features)/2)/parameters["t_obs"])
    features_header = [
        sample_id,
        nb_objects,
        # ids_list,
        parameters["t_obs"],
        parameters["t_pred"]
        # start,
        # stop,
        # parameters["scene"],
        # parameters["framerate"]

    ]

    labels_header = [
        sample_id
    ]
    features = features_header + features
    labels = labels_header + labels

    sample_id += 1

    data_writer.writerow(features)
    label_writer.writerow(labels)
    scene_writer.writerow([scene])
#
    return sample_id
def are_frames_continuous(frames):
    continuous = True
    nb_frames = (float(frames[-1]-frames[0]))/1.0 + 1.0
    if nb_frames > len(frames):
        continuous = False
    return continuous

"""
    parameters: dict containing reauired parameters[
        "original_file" path for file containing the original data
        "frames_temp"   path to store temporarily the frame-shaped data extracted from original_file
        "trajectories_temp" path to store temporarily the trajectory-shaped data extracted from original_file
        "shift" the size of the step between two feature extraction for the main trajectory
        t_obs: number of observed frames
        t_pred: number of frames to predict
        "scene" scene name
        "framerate" framerate of the original data
        "data_path path for file where to write down features
        "label_path" path for file where to write down labels
    ]
"""
def extract_data(parameters):

    helpers.extract_frames(parameters["original_file"],parameters["frames_temp"],save = True)
    helpers.extract_trajectories(parameters["original_file"],parameters["trajectories_temp"],save = True)

    sample_id = 0
    if os.path.exists(parameters["data_path"]):
        os.remove(parameters["data_path"])
    if os.path.exists(parameters["label_path"]):
        os.remove(parameters["label_path"])
    if os.path.exists(parameters["scene_path"]):
        os.remove(parameters["scene_path"])

    with open(parameters["data_path"],"a") as data_csv:
        data_writer = csv.writer(data_csv)
        with open(parameters["label_path"],"a") as label_csv:
            label_writer = csv.writer(label_csv)
            with open(parameters["scene_path"],"a") as scene_csv:
                scene_writer = csv.writer(scene_csv)
                with open(parameters["trajectories_temp"]) as trajectories:
                    with open(parameters["frames_temp"]) as file_frames:
                        for k,trajectory in enumerate(trajectories):

                            
                            trajectory = json.loads(trajectory)

                            scene_name = trajectory["scene"]
                            file_frames,a = tee(file_frames)
                            
                            frames = trajectory["frames"]
                            current_id = int(trajectory["id"])
                            # if current_id == 63:
                            #     print("63")
                            continuous = are_frames_continuous(frames)

                            if continuous:
                                start,stop = frames[0],frames[-1] + 1

                                ids = get_neighbors(islice(a,start,stop))
                                len_traj = len(ids[current_id])

                                for i in range(0,len_traj,parameters["shift"]):
                                    
                                    features,labels = features_labels(len_traj,parameters["shift"],parameters["t_obs"],parameters["t_pred"],current_id,ids,i)
                                    if features != []:
                                        sample_id = persist_data(ids,parameters,current_id,i,features,labels,data_writer,label_writer,scene_writer,sample_id,scene_name)
                                # if sample_id == 39:
                                #     print("")
                            else:
                                print("trajectory {} discarded".format(current_id))
    
    os.remove(parameters["frames_temp"])
    os.remove(parameters["trajectories_temp"])



def main():
    parameters = "parameters/prepare_training.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    print(parameters["shift"])
    
    s = time.time()
    extract_data(parameters)        
    print(time.time()-s)


                    
if __name__ == "__main__":
    main()