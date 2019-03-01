import csv
import os
import time
import extractors.helpers as helpers
import numpy as np
import json
from sklearn import neural_network,preprocessing,metrics,model_selection
import pandas as pd 
import sys



class NgsimExtractor():
    def __init__(self, data_path,param_path):
        
        'Initializing parameters'
        data = json.load(open(data_path))
        param = json.load(open(param_path))

        self.dataset = param["dataset"]
        self.destination_dataset = data["extracted_datasets"]
        self.data_file = data["original_datasets"] +param["data_file"]
        self.user_types = param["user_types"] 
        self.scene_names = param["scene_names"] # list of scenes I want  to keep 
        self.subscene_names = param["subscene_names_all"] # list of subscenes I want  to keep
        self.correspondences = param["correspondences"]
        self.timezones = param["timezones"]
        self.temp = data["temp"]
        self.framerate_ms = param["framerate_ms"]
        self.feet_meters = param["feet_meters"]
        self.destination_file = data["extracted_datasets"] + "{}.csv"
        self.temp_destination_file = data["temp"] + "{}.csv"
        self.clips = param["clip_scenes"]


    def extract(self):
    
        print("Clean temp dir")
        helpers.del_files_containing_string(self.scene_names,self.temp) 
        print("Done!")
        print("split ngsim observations in subfiles according to scene corresponences")
        self.__split_ngsim_correspondences()
        print("Done!")


        print("Clean destination dir")
        helpers.del_files_containing_string(self.scene_names,self.destination_dataset) 
        print("Done!")

        print("Extract split observations to destination file")

        trajectories = {}
        self.trajectory_counter = 0
        for subscene in self.subscene_names:
            print("------ Processing {}".format(subscene))
            data_path = self.temp_destination_file.format(subscene)
            with open(data_path) as data_reader:
                data_reader = csv.reader(data_reader, delimiter=',')
                last_id = -1
                for line in data_reader:
                    new_id = int(line[0])
                    if  last_id != new_id:   
                        trajectories = self.__persist_trajectories(trajectories,subscene)
                        self.trajectory_counter += 1
                    self.__add_obs(trajectories,line,subscene)
                    last_id = new_id

            trajectories = self.__persist_trajectories(trajectories,subscene)
        print("Done!")
        print("Clean temp dir")
        helpers.del_files_containing_string(self.scene_names,self.temp) 
        print("Done!")

    def __clip_scene(self):
        for subscene in self.clips: 
            helpers.clip_scene(self.clips[subscene],self.destination_file.format(subscene))
    
    def __get_bbox(self,direction,new_pos,length,width):
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

    """
        add the new observation to one of the current trajectories
        or to a new one
        converts pos unit from feets to meters
        compute the current frame of the observation based on the min global (in timezones) epoch of
        the scene
    """  
    def __add_obs(self,trajectories,row,subscene):
        id_,frame,type_ = int(row[0]),int(row[1]),self.user_types[row[10]]

        time_epoch = int(row[3])

        # min value for epoch in the subscene
        min_epoch = self.timezones[subscene]

        # don'T get the local frame, compute current frame from global time
        frame = int( (time_epoch - min_epoch)/ self.framerate_ms )
        id_ = self.trajectory_counter
        new_pos = [
                    self.feet_meters*float(row[4]),
                    self.feet_meters*float(row[5])
                ]
    

        # width and length of the observed vehicle
        length = self.feet_meters*float(row[8])
        width = self.feet_meters*float(row[9])

        direction = int(row[18])
        bbox = self.__get_bbox(direction,new_pos,length,width)


        if id_ not in trajectories:

            trajectories[id_] = {
                "coordinates":[new_pos],
                "frames":[frame],
                "type": type_,
                "subscene" : subscene,
                "bboxes" : [bbox],
                "dataset": self.dataset,                                                        
                }
        else:
            

            trajectories[id_]["coordinates"].append(new_pos)
            trajectories[id_]["frames"].append(frame)
            trajectories[id_]["bboxes"].append(bbox)
        

    """
        get a dict of the completed trajectories
        add each trajectory observation in the given filepath
        returns an empy dict
    """
    def __persist_trajectories(self,trajectories,subscene):
        file_path = self.destination_file.format(subscene)
        for id_ in trajectories:
            
            trajectory = trajectories[id_]
            subscene = trajectory["subscene"]
            type_ = trajectory["type"]
            
            with open(file_path,"a") as csv_:
                csv_writer = csv.writer(csv_)
                
                for frame,pos,bbox in zip(trajectory["frames"],trajectory["coordinates"],trajectory["bboxes"]):
                    
                    row = self.__parse_row(subscene,frame,id_,pos,bbox,type_)
                    csv_writer.writerow(row)

        return {}

    def __parse_row(self,subscene,frame,id_,pos,bbox,type_):
        row = []
        row.append(self.dataset) #dataset
        row.append(subscene) #scene
        row.append(frame) # frame
        row.append(id_) # id
        row.append(pos[0]) #x
        row.append(pos[1])  #y
        for b in bbox:
            row.append(b) 
        row.append(type_)
        return row

                    
    """
        data_file: the ngsim file containing every observation across multiple scenes
        correspondences: {
            "scene i": {
                inters: {
                    inter_id_i: ["new_Scene_name"]
                }
                sections: {
                    section_id_i: "new_Scene_name
                }
            }
        } i.e. for a given scene and a given intersection stores the new scene in which to add this
        observation
        temp: directory to store the files of the new scenes
        go through ngsim file and store observations if belonging to selected scenes
    """
    def __split_ngsim_correspondences(self):
        with open(self.data_file) as data_reader:
            data_reader = csv.reader(data_reader, delimiter=',')
            for i, line in enumerate(data_reader):               
                if i != 0:

                    # get scene name (i.e. lankershim or peachtree)
                    scene = line[-1]
                    
                    # if the scene is one that we keep
                    if scene in self.correspondences:
                        corres_scene = self.correspondences[scene]
                        inter = line[16]
                        section = line[17]

                        # either inter or section is 0
                        # if one is different from 0, get the scenes
                        # where to add this observation
                        if inter in corres_scene["inters"]:
                            subscenes = corres_scene["inters"][inter]
                        elif section in corres_scene["sections"]:
                            subscenes = corres_scene["sections"][section]
                        else:
                            subscene = []


                        # write this observation in the selected scenes
                        for subscene in subscenes:
                            file_path = self.temp_destination_file.format(subscene)
                            with open(file_path,"a") as csv_:
                                csv_writer = csv.writer(csv_)
                                csv_writer.writerow(line)
 
# python ngsim_extractor.py parameters/data.json parameters/ngsim_extractor.json
    
def main():
    args = sys.argv
    ngsim_extractor = NgsimExtractor(args[1],args[2])
    ngsim_extractor.extract()

    # for subscene in clips:
    #     file_path = csv_path + subscene 
    #     print(file_path)
    #     clip = clips[subscene]
    #     print(clip)
    #     clip_scene.clip_scene(clip[0],clip[1],clip[2],clip[3],file_path)


                            
                                
if __name__ == "__main__":
    main()
