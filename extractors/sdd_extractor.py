import time
import csv
import helpers
import os
import sys
import json

class SddExtractor():
    def __init__(self, data_path,param_path):
        
        'Initializing parameters'
        data = json.load(open(data_path))
        param = json.load(open(param_path))

        self.dataset = param["dataset"]
        self.original_dataset = data["original_datasets"] + param["data_path"]
        self.destination_file = data["extracted_datasets"] + "{}" + "{}" + ".csv"
        self.user_types = param["user_types"]
        self.scene_dirs = param["scene_dirs"]
        self.data_path = param["data_path"]
        self.scene_path = data["original_datasets"] + param["data_path"] + "{}"
        self.subscene_path = data["original_datasets"] + param["data_path"] + "{}/{}" + param["trajectory_file"]
          
    def extract(self):       
        for scene in self.scene_dirs:
            print("Processing scene: " + scene)
            subscene_names = helpers.get_dir_names(self.scene_path.format(scene))            
            for i,sub in enumerate(subscene_names):
                print("------subscene: " + scene + str(i))                
                subscene_path = self.subscene_path.format(scene,sub)
                scene_csv = self.destination_file.format(scene,i)
                helpers.remove_file(scene_csv)

                with open(scene_csv,"a") as csv_scene:
                    writer_scene = csv.writer(csv_scene)   
                    with open(subscene_path) as subscene_csv:
                        csv_reader = csv.reader(subscene_csv)                    
                        for row in csv_reader:
                            new_row = self.__parse_row(row,scene + str(i), self.dataset,self.user_types )                            
                            writer_scene.writerow(new_row)
    def __parse_row(self,row,scene, dataset,dict_type):
        row = row[0].split(" ")
        new_row = []    
        
        xa = float(row[1])
        ya = float(row[2])
        xb = float(row[3])
        yb = float(row[4])

        x = str((xa + xb)/2)
        y = str((ya+yb)/2 )

        new_row.append(dataset) # dataset label
        new_row.append(scene)   # subscene label
        new_row.append(row[5]) #frame
        new_row.append(row[0]) #id

        new_row.append(x) #x
        new_row.append(y) #y
        new_row.append(row[1]) # xmin. The top left x-coordinate of the bounding box.
        new_row.append(row[2]) # ymin The top left y-coordinate of the bounding box.
        new_row.append(row[3]) # xmax. The bottom right x-coordinate of the bounding box.
        new_row.append(row[4]) # ymax. The bottom right y-coordinate of the bounding box.

        new_row.append(dict_type[row[9]]) # label type of agent    

        return new_row
"""
    Parse SDD line by line and return new csv files
    One csv file by scene (including its subscenes)
    One global csv file (including every scene)
"""
# python sdd_extractor.py parameters/data.json parameters/sdd_extractor.json

def main():
    args = sys.argv   
    sdd_extractor = SddExtractor(args[1],args[2])
    sdd_extractor.extract()

if __name__ == "__main__":
    main()



#