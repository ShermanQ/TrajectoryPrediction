import sys 
import os
import helpers
import json
import csv


class SamplesManager():
    def __init__(self,data,params):
        data = json.load(open(data))
        params = json.load(open(params))


        self.original_samples = data["prepared_samples"]
        self.original_labels = data["prepared_labels"] 

        self.dest_samples = data["prepared_samples_grouped"]
        self.dest_labels = data["prepared_labels_grouped"]

        self.dest_ids = data["prepared_ids"]

        self.scene_list = params["selected_scenes"]

    def regroup(self):
        print("Grouping samples")
        self.__concatenate_csv(self.original_samples,self.dest_samples,False)
        print("Grouping labels")

        self.__concatenate_csv(self.original_labels,self.dest_labels,True)


    def __concatenate_csv(self,files_dir,destination_file,save_ids):
        helpers.remove_file(destination_file)
        helpers.remove_file(self.dest_ids)

        
        ids = []
        with open(destination_file,"a") as dest_csv:
            

            dest_writer = csv.writer(dest_csv)
            

            for i,file_ in enumerate(self.scene_list):
                file_csv = files_dir + file_ +".csv"
                with open(file_csv) as file_csv:
                    file_reader = csv.reader(file_csv)

                    for k,row in enumerate(file_reader):
                        dest_writer.writerow(row)
                        id_ = file_ +"_"+ str( i * 1000000 +k)
                        ids.append(id_)
            if save_ids:
                with open(self.dest_ids,"a") as ids_json:
                    json.dump({"ids":ids},ids_json)