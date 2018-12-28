import numpy as np 
import csv
import os
import time

DATASETS = "./datasets/"
DATASET = "ngsim"
DATAFILE = DATASETS + DATASET + "/" + DATASET + ".csv"
CSV = "./csv/"

def get_scene_names(filename):
    line_num = 0
    scene_names = ["names: "]
    with open(filename) as csv_reader:
        for line in csv_reader:
            if line_num != 0:
                scene_name = line.split(",")[-1]
                if scene_name not in scene_names:
                    scene_names.append(scene_name)
                
            line_num += 1
    return [s[:-1] for s in scene_names[1:]]

def main():

    print("checking the existence of destination files...")
    dict_type = {1:"motorcycle", 2:"car", 3:"truck"}
    scene_names = get_scene_names(DATAFILE)

    for scene_name in scene_names:
        scene_file = CSV + scene_name + ".csv"
        if os.path.exists(scene_file):
            os.remove(scene_file)
    print("Done!")
    # print(scene_names)
  
    # for scene_name in scene_names:
    #     scene_file = CSV + scene_name + ".csv"
    #     print("processing scene: " + str(scene_name))

        # with open(scene_file,"a") as scene_csv:
        #     scene_writer = csv.writer(scene_csv)
    print("processing extraction...")

    # print_every = 3000000
    start = time.time()
    
        
    for scene_name in scene_names:
        print("processing scene: " + scene_name)
        with open(DATAFILE) as csv_reader:
   
            
            scene_file = CSV + scene_name + ".csv"

            with open(scene_file,"a") as scene_csv:
                scene_writer = csv.writer(scene_csv)

                line_num = 0

                for line in csv_reader:
                    # if line_num % print_every == 0:
                    #     print(line_num)
                    #     print(time.time()-start)

                    if line_num != 0:
                        line = line.split(",")
                        scene_line = line[-1][:-1]

                        if scene_name == scene_line:


                            row = []
                            row.append(DATASET) #dataset
                            row.append(scene_name) #scene
                            row.append(int(line[1])) # frame
                            row.append(int(line[0])) # id
                            row.append(float(line[4])) #x
                            row.append(float(line[5])) #y

                            row.append(float(0)) #xl
                            row.append(float(0)) #yl
                            row.append(float(0)) #xb
                            row.append(float(0)) #yb

                            row.append(dict_type[int(line[10])]) # type

                            scene_writer.writerow(row)



                    line_num += 1
    print(time.time()-start)
    



if __name__ == "__main__":
    main()