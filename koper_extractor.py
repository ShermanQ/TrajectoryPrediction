import scipy.io as sio
import numpy as np 
import extractors.helpers as helpers
import csv
import os 
import sys
import json
#TODO self.get_bb

class KoperExtractor():
    def __init__(self, data_path,param_path):
        
        'Initializing parameters'
        data = json.load(open(data_path))
        param = json.load(open(param_path))

        self.dataset = param["dataset"]
        self.original_dataset = data["original_datasets"] + param["dataset"] + "/"
        self.destination_file = data["extracted_datasets"] + param["dest_file"]
        self.user_types = param["user_types"]
    
    
    def extract(self):
        files_name = helpers.get_dir_names(self.original_dataset)
        helpers.remove_file(self.destination_file)

        with open(self.destination_file,"a") as csv_file:
            csv_writer = csv.writer(csv_file)
            frame = 0 

            for file_index,file_ in enumerate(files_name):
                current_file = sio.loadmat(self.original_dataset + file_)

                result = current_file["result"]
                boxes = result["BoxfittingLabels_REF"][0][0]
                data_ref = boxes["data"][0][0][0]

                for ref in data_ref:
                    ref = ref["features"][0][0]

                    for object_ in ref:
                        row = self.__get_row(object_,frame,file_index)
                        csv_writer.writerow(row)           
                    frame += 1

    

    def __get_row(self,object_,frame,file_index):
        theta = float(object_[6])
        width = float(object_[3])
        length = float(object_[4])
        x = float(object_[0])
        y = float(object_[1])
        # t,b,pm = helpers.get_bbox(theta,width,length,x,y)
        t = b = [-1,-1]
        pm = [x,y]

        row = []
        row.append(self.dataset) #dataset
        row.append(self.dataset) #scene
        row.append(frame)   #frame
        row.append(int(object_[-2])+ file_index*1000)  # object_id
        
        row.append(pm[0]) #x
        row.append(pm[1]) #y
        row.append(t[0]) #xl
        row.append(t[1]) #yl
        row.append(b[0]) #xb
        row.append(b[1]) #yb
        row.append(self.user_types[str(int(object_[-1]))]) #object type

        return row

# python koper_extractor.py parameters/data.json parameters/koper_extractor.json
def main():
    args = sys.argv
    koper_extractor = KoperExtractor(args[1],args[2])
    koper_extractor.extract()


if __name__ == "__main__":
    main()

