import scipy.io as sio
import numpy as np 
import extractors.helpers as helpers
import csv
import os 

DATA_PATH = "./data/datasets/koper/"
DATA_FILES = "/annotations.txt"
DATASET = "koper"
CSV_PATH = "./data/csv/"



def main():

    dict_type = {0: "pedestrian", 1: "bicycle", 3: "car", 5: "truck" }
    files_name = helpers.get_dir_names(DATA_PATH)
    print(files_name)
   

    file_path = CSV_PATH + DATASET + ".csv"

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path,"a") as csv_file:
        csv_writer = csv.writer(csv_file)


        frame = 0 

        
        # ids = []

        # files_name = ["sequence1a.mat"]
        for i,file_ in enumerate(files_name):
            # dict_ = {}
            
            
            current_file = sio.loadmat(DATA_PATH + file_)

            result = current_file["result"]
            boxes = result["BoxfittingLabels_REF"][0][0]
            data_ref = boxes["data"][0][0][0]

            for ref in data_ref:
                ref = ref["features"][0][0]

                for object_ in ref:
                    # a = int(object_[-2])
                    # # if int(object_[-2]) not in ids:
                    # if a in dict_:
                    #     dict_[a] += 1
                    # else:
                    #     dict_[a] = 1
                        # ids.append(int(object_[-2]))
                    theta = float(object_[6])
                    width = float(object_[3])
                    length = float(object_[4])
                    x = float(object_[0])
                    y = float(object_[1])
                    t,b,pm = helpers.get_bbox(theta,width,length,x,y)

                    row = []
                    row.append(DATASET) #dataset
                    row.append(DATASET) #scene
                    row.append(frame)   #frame
                    row.append(int(object_[-2])+ i*1000)  # object_id
                    
                    row.append(pm[0]) #x
                    row.append(pm[1]) #y
                    row.append(t[0]) #xl
                    row.append(t[1]) #yl
                    row.append(b[0]) #xb
                    row.append(b[1]) #yb
                    row.append(dict_type[int(object_[-1])]) #object type
                    csv_writer.writerow(row)
                    

                frame += 1

if __name__ == "__main__":
    main()

