import csv
from itertools import islice
import extractors.helpers as helpers 
import json
import numpy as np
import time

"""
    in: 
        start:frame number where the trajectory begins
        stop: frame number where the trajectory stops
        frames_path: path to the file containing frames
    out:
        dict: {id0:[], ..., idn:[]} ids of vehicle in the sequence

"""

def get_neighbors_id(start,stop,frames_path):
    ids = {}
    with open(frames_path) as frames:
        
        for frame in islice(frames,start,stop):
            frame = json.loads(frame)
            for id_ in frame:
                if id_ != "frame":
                    if int(id_) not in ids:
                        ids[int(id_)] = []
    return ids

"""
    in: 
        start:frame number where the trajectory begins
        stop: frame number where the trajectory stops
        frames_path: path to the file containing frames
        ids: {id0:[], ..., idn:[]} ids of vehicle in the sequence
    out:
        returns ids but the lists are filled with the coordinates
        of theirs objects for a given frame or [-1,-1] if its not in 
        the given frame
"""
def get_neighbors_coordinates(start,stop,frames_path,ids):
    with open(frames_path) as frames:
        for frame in islice(frames,start,stop):
            frame = json.loads(frame)

            for id_ in ids:
                if str(id_) in frame:
                    ids[id_].append(frame[str(id_)]["coordinates"])
                else:
                    ids[id_].append([-1,-1])
    return ids

"""
    in:
        ids and corresponding coordinates of vehicle present
        in a subsequence of the scene
        current_id: id of the main object
    out:
        coordinates data, one column per object, the main object
        is in first column
        the others columns are sorted by ascending objects
"""
def neighbors_to_data(ids,current_id):
    data = [ids[current_id]]
    for id_ in sorted(ids):
        if id_ != current_id:
            data.append(ids[id_])
    return data
def main():
    parameters = {
        "original_file" : "./data/csv/new_rates/lankershim_section2_10.0to2.5.csv",
        "frames_temp" : "./data/temp/frames.txt",
        "trajectories_temp" : "./data/temp/trajectories.txt",
        "shift": 4,
        "t_obs": 8,
        "t_pred": 12,
        "scene": "lankershim_inter2",
        "framerate" : "2.5",
        "data_path": "./data/deep/data.csv",
        "label_path": "./data/deep/labels.csv",


    }
    
    s = time.time()
    helpers.extract_frames(parameters["original_file"],parameters["frames_temp"],save = True)

    print(time.time()-s)
    helpers.extract_trajectories(parameters["original_file"],parameters["trajectories_temp"],save = True)

    print(time.time()-s)

    sample_id = 0
    with open(parameters["data_path"],"a") as data_csv:
        data_writer = csv.writer(data_csv)
        with open(parameters["label_path"],"a") as label_csv:
            label_writer = csv.writer(label_csv)

            with open(parameters["trajectories_temp"]) as trajectories:
                for k,trajectory in enumerate(trajectories):
                    trajectory = json.loads(trajectory)
                    frames = trajectory["frames"]
                    current_id = int(trajectory["id"])
                    start,stop = frames[0],frames[-1] + 1

                    
                    ids = get_neighbors_id(start,stop,parameters["frames_temp"])
                    ids = get_neighbors_coordinates(start,stop,parameters["frames_temp"],ids)
                    data = neighbors_to_data(ids,current_id)
                    data = np.array(data)
                    
                    
                    for i in range(0,len(data[0]),int(parameters["shift"])):
                        features = []
                        labels = []
                        if i + parameters["t_obs"] + parameters["t_pred"] -1 < len(data[0]):
                            for j in range(len(data)):
                                feature = data[j,i: i + parameters["t_obs"] ].flatten()
                                label = data[j,i + parameters["t_obs"]: i + parameters["t_obs"] + parameters["t_pred"] ].flatten()

                                features.append(feature.tolist())
                                labels.append(label.tolist())
                        features = np.array(features).flatten().tolist()
                        labels = np.array(labels).flatten().tolist()

                        ids = sorted([id_ for id_ in ids])
                        nb_objects = len(ids)
                        # ids = sorted(ids.keys())
                        features_header = [
                            sample_id,
                            nb_objects,
                            ids,
                            parameters["t_obs"],
                            parameters["t_pred"],
                            start,
                            stop,
                            parameters["scene"],
                            parameters["framerate"]

                        ]

                        labels_header = [
                            sample_id
                        ]
                        features = features_header + features
                        labels = labels_header + labels

                        sample_id += 1

                        data_writer.writerow(features)
                        label_writer.writerow(labels)

                    print(k,sample_id,time.time()-s)

    print(time.time()-s)


                    
if __name__ == "__main__":
    main()