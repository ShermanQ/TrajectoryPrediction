import numpy as np 
import csv

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
    # scene_names = get_scene_names(DATAFILE)
    # print(scene_names)
  
    # for scene_name in scene_names:
    #     scene_file = CSV + scene_name + ".csv"
    #     print("processing scene: " + str(scene_name))

        # with open(scene_file,"a") as scene_csv:
        #     scene_writer = csv.writer(scene_csv)

    with open(DATAFILE) as csv_reader:
        line_num = 0
        for line in csv_reader:
            if line_num != 0:
                line = line.split(",")
                scene_name = line[-1][:-1]
                scene_file = CSV + scene_name + ".csv"

                print(scene_file)
            line_num += 1
    



if __name__ == "__main__":
    main()