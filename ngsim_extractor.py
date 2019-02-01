import csv
import os
import time
import extractors.helpers as helpers
import numpy as np
import json
from sklearn import neural_network,preprocessing,metrics,model_selection
import pandas as pd 
import clip_scene


def del_files_containing_string(strings,dir_path):
    # in data/csv delete every csv file related to the selected scene
    csv_files = helpers.get_dir_names(dir_path)
    for csv_ in csv_files:
        for string in strings:
            if string in csv_:
                file_ = dir_path + csv_
                # print(scene_file)
                if os.path.exists(file_):
                    os.remove(file_)

def feet_meters(value,conversion_rate = 0.3048 ):
    return conversion_rate * value

def get_bbox(direction,new_pos,length,width):
    if direction == 2:
        top_left = np.subtract(new_pos, [width/2.,length]).tolist()
        bottom_right = np.subtract(new_pos, [-width/2.,0]).tolist()
    elif direction == 4:
        top_left = np.subtract(new_pos, [width/2.,0]).tolist()
        bottom_right = np.subtract(new_pos, [-width/2.,-length]).tolist()
    elif direction == 3 :
        top_left = np.subtract(new_pos, [0.,-width/2]).tolist()
        bottom_right = np.subtract(new_pos, [-length,width/2]).tolist()
    elif direction == 1 :
        top_left = np.subtract(new_pos, [length,-width/2]).tolist()
        bottom_right = np.subtract(new_pos, [0,width/2]).tolist()
    else: 
        top_left,bottom_right = [-1,-1],[-1,-1]
    return [c for c in top_left+bottom_right]
    
def add_obs(trajectories,row,subscene,dataset,dict_type,trajectory_counter,timezones,rate = 100.0):
    id_,frame,type_ = int(row[0]),int(row[1]),dict_type[row[10]]

    time_epoch = int(row[3])
    min_epoch = timezones[subscene]
    frame = int( (time_epoch - min_epoch)/ rate )
    id_ = trajectory_counter
    new_pos = [
                feet_meters(float(row[4])),
                feet_meters(float(row[5]))
            ]
   

    # width and length of the observed vehicle
    length = feet_meters(float(row[8]))
    width = feet_meters(float(row[9]))

    direction = int(row[18])
    bbox = get_bbox(direction,new_pos,length,width)


    if id_ not in trajectories:

        trajectories[id_] = {
            "coordinates":[new_pos],
            "frames":[frame],
            "type": type_,
            "subscene" : subscene,
            "bboxes" : [bbox],
            "dataset": dataset,                                                        
            }
    else:
        

        trajectories[id_]["coordinates"].append(new_pos)
        trajectories[id_]["frames"].append(frame)
        trajectories[id_]["bboxes"].append(bbox)
    return trajectory_counter

def persist_trajectories(trajectories,file_path):
    for id_ in trajectories:
        
        trajectory = trajectories[id_]
        dataset = trajectory["dataset"]
        subscene = trajectory["subscene"]
        type_ = trajectory["type"]
        
        file_path +=  ".csv"

        with open(file_path,"a") as csv_:
            csv_writer = csv.writer(csv_)

            
            for frame,pos,bbox in zip(trajectory["frames"],trajectory["coordinates"],trajectory["bboxes"]):
                
                row = []
                row.append(dataset) #dataset
                row.append(subscene) #scene
                row.append(frame) # frame
                row.append(id_) # id
                row.append(pos[0]) #x
                row.append(pos[1])  #y
                for b in bbox:
                    row.append(b) 
                row.append(type_)
                csv_writer.writerow(row)

    return {}

def split_ngsim(data_file):
    with open(data_file) as data_reader:
        data_reader = csv.reader(data_reader, delimiter=',')
        for i, line in enumerate(data_reader):               
            if i != 0:
                if line[16] != "0" and line[17] == "0":
                    subscene = line[-1] +"_inter" +line[16]
                    file_path = "./data/datasets/ngsim/" + subscene + ".csv"
                    with open(file_path,"a") as csv_:
                        csv_writer = csv.writer(csv_)
                        csv_writer.writerow(line)


                
                



                

                


def split_ngsim_correspondences(data_file,correspondences):
    with open(data_file) as data_reader:
        data_reader = csv.reader(data_reader, delimiter=',')
        for i, line in enumerate(data_reader):               
            if i != 0:
                scene = line[-1]
                
                if scene in correspondences:
                    corres_scene = correspondences[scene]
                    inter = line[16]
                    section = line[17]

                    if inter in corres_scene["inters"]:
                        subscenes = corres_scene["inters"][inter]
                    elif section in corres_scene["sections"]:
                        subscenes = corres_scene["sections"][section]
                    else:
                        subscene = []


                    
                    for subscene in subscenes:
                        file_path = "./data/datasets/ngsim/" + subscene + ".csv"
                        with open(file_path,"a") as csv_:
                            csv_writer = csv.writer(csv_)
                            csv_writer.writerow(line)
 
    
def main():
    parameters = "parameters/ngsim_extractor.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    csv_path = "./data/csv/"
    dataset = parameters["dataset"]
    data_file = parameters["data_file"]
    data_dir = parameters["data_dir"]
    dict_type = parameters["dict_type"]    
    scene_names = parameters["scene_names"] # list of scenes I want  to keep    
    subscene_names = parameters["subscene_names_all"] # list of subscenes I want  to keep
    correspondences = parameters["correspondences"]
    # models_json = parameters["models"]
    clips = parameters["clip_scenes"]

    timezones = parameters["timezones"]

    start = time.time()

    
    # del_files_containing_string(scene_names,data_dir) 
    # split_ngsim_correspondences(data_file,correspondences)
  
 

    print(time.time() - start)
    # split_ngsim(data_file)
    del_files_containing_string(scene_names,csv_path) 

    trajectories = {}
    trajectory_counter = 0

    
    for subscene in subscene_names:
        data_path = "./data/datasets/ngsim/"+subscene+".csv"
        with open(data_path) as data_reader:
            data_reader = csv.reader(data_reader, delimiter=',')

            last_id = -1
            for line in data_reader:
            
                
                new_id = int(line[0])
                if  last_id != new_id:

                    
                    file_path = csv_path + subscene 
                    

                    
                    trajectories = persist_trajectories(trajectories,file_path)
                    trajectory_counter += 1
                    

                trajectory_counter = add_obs(trajectories,line,subscene,dataset,dict_type,trajectory_counter,timezones)
                last_id = new_id

        file_path = csv_path + subscene + ".csv"
        trajectories = persist_trajectories(trajectories,file_path)

    for subscene in clips:
        file_path = csv_path + subscene 
        print(file_path)
        clip = clips[subscene]
        print(clip)
        clip_scene.clip_scene(clip[0],clip[1],clip[2],clip[3],file_path)


                            
                                

    print(time.time()-start)
if __name__ == "__main__":
    main()
