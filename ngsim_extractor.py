import csv
import os
import time
import extractors.helpers as helpers
import numpy as np
import json

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

def add_obs(trajectories,row,subscene,dataset,dict_type):
    id_,frame,type_ = int(row[0]),int(row[1]),dict_type[row[10]]
    new_pos = [
                feet_meters(float(row[4])),
                feet_meters(float(row[5]))
            ]

    top_left,bottom_right = [-1,-1],[-1,-1]

    if id_ not in trajectories:

        trajectories[id_] = {
            "coordinates":[new_pos],
            "frames":[frame],
            "type": type_,
            "subscene" : subscene,
            "bboxes" : [[c for c in top_left+bottom_right]],
            "dataset": dataset                                                        
            }
    else:
        trajectories[id_]["coordinates"].append(new_pos)
        trajectories[id_]["frames"].append(frame)
        trajectories[id_]["bboxes"].append([c for c in top_left+bottom_right])

def persist_trajectories(trajectories,file_path):
    with open(file_path,"a") as csv_:
        csv_writer = csv.writer(csv_)

        for id_ in trajectories:
            trajectory = trajectories[id_]
            dataset = trajectory["dataset"]
            subscene = trajectory["subscene"]
            type_ = trajectory["type"]

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

    
def main():
    parameters = "parameters/ngsim_extractor.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    csv_path = "./data/csv/"
    dataset = parameters["dataset"]
    # data_file = parameters["data_file"]
    # feet_meters = parameters["feet_meters"]
    dict_type = parameters["dict_type"]    
    scene_names = parameters["scene_names"] # list of scenes I want  to keep    
    subscene_names = parameters["subscene_names_all"] # list of subscenes I want  to keep


    start = time.time()
    trajectories = {}
    
  

    del_files_containing_string(scene_names,csv_path) 

    
    for subscene in subscene_names:
        data_path = "./data/datasets/ngsim/"+subscene+".csv"
        with open(data_path) as data_reader:
            data_reader = csv.reader(data_reader, delimiter=',')

            last_id = -1
            for i, line in enumerate(data_reader):
              
                
                new_id = int(line[0])
                if  last_id != new_id:
                    file_path = csv_path + subscene + ".csv"
                    
                    trajectories = persist_trajectories(trajectories,file_path)
                    

                add_obs(trajectories,line,subscene,dataset,dict_type)
                last_id = new_id
                            
                            

                            
                                

    print(time.time()-start)
if __name__ == "__main__":
    main()