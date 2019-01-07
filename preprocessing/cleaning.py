import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories

import time
import json
import numpy as np
from scipy.spatial.distance import euclidean

ROOT = "./../"
CSV = ROOT + "extractors/csv/"


def missing_values(file_path,nb_features = 11):

    nb_missing = 0

    nb_missing_features = [ 0 for _ in range(nb_features)]
    lines = []
    with open(file_path) as csv:

        for i,line in enumerate(csv):
            line = line.split(",")

            for k,e in enumerate(line):

                if e == "":
                    nb_missing_features[k] += 1


            
            if "" in line:
                nb_missing += 1
                lines.append(i)
                print(line)
            
            
    return nb_missing,nb_missing_features,lines

def missing_values_multiple(file_pathes,nb_features = 11,func = missing_values):


    for csv in file_pathes:
        print("searching for missing values in: " + csv)
        nb_missing,nb_missing_features,lines = func(csv,nb_features)
        print("number of missing values: " + str(nb_missing))
        print("number of missing values per features: " + str(nb_missing_features))
        print("lines where nan occured: " + str(lines))



def collisions_in_frame(frame,threshold = 0.1):

    conflicts = []

    old_ids = []
    nb_objects = 0
    ious = []
    ious_conflict = []
    for key_id1 in frame:
        
        if key_id1 != "frame":
            nb_objects += 1
            object1 = frame[key_id1]
            for key_id2 in frame:
                if key_id1 != key_id2 and key_id2 != "frame" and key_id2 not in old_ids:
                    object2 = frame[key_id2]
                    iou = bb_intersection_over_union(object1["bbox"], object2["bbox"])
                    ious.append(iou)

                    if iou > threshold:
                        ious_conflict.append(iou)
                        conflict = {
                            "frame": frame["frame"], 
                            "id1":key_id1,
                            "id2": key_id2, 
                            "iou": iou
                            } 
                        conflicts.append(conflict)
            old_ids.append(key_id1)
    nb_collisions = len(conflicts)
    return conflicts,nb_collisions,frame["frame"],nb_objects,ious,ious_conflict


def collisions_in_scene(file_path,temp_path = "./temp.txt"):
    # to be called in notebook with dataframe describe for further analysis
    extract_frames(file_path,temp_path,save=True)
    nb_collisions_total = []
    nb_objects_total = []
    ious_total = []
    ious_conflict_total = []
    with open(temp_path) as frames:
        for frame in frames:
            frame = json.loads(frame)
            # print(frame["0"]["bbox"])
            conflicts,nb_collisions,frame_id,nb_objects,ious,ious_conflict = collisions_in_frame(frame,threshold = 0.1)
            nb_collisions_total.append(nb_collisions)
            nb_objects_total.append(nb_objects)

            for iou in ious:
                ious_total.append(iou)
            for iou in ious_conflict:
                ious_conflict_total.append(iou)
    os.remove(temp_path)

    return nb_collisions_total,nb_objects_total,ious_total,ious_conflict_total

    # print(np.mean(nb_collisions_total),np.std(nb_collisions_total))
    # print(np.mean(nb_objects_total),np.std(nb_objects_total))
    # print(np.mean(ious_total),np.std(ious_total))
    # print(np.mean(ious_conflict_total),np.std(ious_conflict_total))

def trajectories_continuity(file_path,temp_path = "./temp.txt"):
    # to be called in notebook with dataframe describe for further analysis
    extract_trajectories(file_path,temp_path, save = True)
    trajectories_deltas = {}
    trajectories_length = {}
    with open(temp_path) as trajectories:
        for trajectory in trajectories:
            trajectory = json.loads(trajectory)
            deltas = trajectory_continuity(trajectory,threshold = 0.1)
            trajectories_deltas[trajectory["id"]] = deltas
            trajectories_length[trajectory["id"]] = len(trajectory["coordinates"])
    os.remove(temp_path)
    return trajectories_deltas,trajectories_length


def trajectory_continuity(trajectory,threshold = 0.1):

    coordinates = trajectory["coordinates"]
    deltas = []
    for i in range(1,len(coordinates)):
        delta = euclidean(coordinates[i],coordinates[i-1])
        deltas.append(delta)

    return deltas

    # print(trajectory["id"],np.mean(deltas),np.std(deltas))
        


def main():
    csvs = [ CSV + f for f in get_dir_names(CSV,lower = False) if f != "main"]
    # s = time.time()   

    csv = csvs[1]
    # for csv in csvs:
    #     print(csv)
    #     deltas,lengths = trajectories_continuity(csv,temp_path = "./temp.txt")
    # for csv in csvs:
    #     print(csv)
    #     nb_collisions_total,nb_objects_total,ious_total,ious_conflict_total = collisions_in_scene(csv) 
    #     print(time.time() - s)               
    # print(time.time() - s)
    # missing_values_multiple(csvs)




if __name__ == "__main__":
    main()