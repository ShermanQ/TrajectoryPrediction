import scipy.io as sio
import numpy as np 
import helpers
import csv
import os 

DATA_PATH = "./datasets/koper/"
DATA_FILES = "/annotations.txt"
DATASET = "koper"
CSV_PATH = "./csv/"

def main():
    files_name = helpers.get_dir_names(DATA_PATH)
    print(files_name)
   

    file_path = CSV_PATH + DATASET + ".csv"

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path,"a") as csv_file:
        csv_writer = csv.writer(csv_file)


        frame = 0 

        for file_ in files_name:
            
            current_file = sio.loadmat(DATA_PATH + file_)

            result = current_file["result"]
            boxes = result["BoxfittingLabels_REF"][0][0]
            data_ref = boxes["data"][0][0][0]

            for ref in data_ref:
                ref = ref["features"][0][0]

                for object_ in ref:

                    row = []
                    row.append(DATASET) #dataset
                    row.append(DATASET) #scene
                    row.append(frame)   #frame
                    row.append(object_[6])  # object_id
                    row.append(object_[0])  #x
                    row.append(object_[1]) #y
                    row.append(-1) #xl
                    row.append(-1) #yl
                    row.append(-1) #xb
                    row.append(-1) #yb
                    row.append(object_[7]) #object type
                    csv_writer.writerow(row)

                frame += 1
    
    print(frame)
        


    # print(len(data))


if __name__ == "__main__":
    main()

