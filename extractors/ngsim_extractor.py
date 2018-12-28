import csv
import os
import time
import helpers

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

    start = time.time()
    print("checking the existence of destination files...")
    dict_type = {1:"motorcycle", 2:"car", 3:"truck"}
    # scene_names = get_scene_names(DATAFILE)
    
    print(time.time()-start)

    scene_names = ["lankershim" ,"peachtree"]

    csv_files = helpers.get_dir_names(CSV)
    for csv_ in csv_files:
        for scene_name in scene_names:
            if scene_name in csv_:
                scene_file = CSV + csv_
                # print(scene_file)
                if os.path.exists(scene_file):
                    os.remove(scene_file)
    print("Done!")

    scene_names = ["lankershim"]
    # print("extracting subscenes id")
    # subscenes = get_subscenes(scene_names)
    # print("Done!")
    # print(time.time()-start)

    print("processing extraction...")

    feet_meters = 0.3048
    for scene_name in scene_names:
        print("processing scene: " + scene_name)
        with open(DATAFILE) as csv_reader:
            line_num = 0

            for line in csv_reader:

                if line_num != 0:
                    line = line.split(",")
                    scene_line = line[-1][:-1]
                    

                    if scene_name == scene_line:

                        new_scene_name = scene_name

                        if line[16] != "0":
                            new_scene_name = scene_name +"_inter" +line[16]                            
                        elif line[17] != "0":
                            new_scene_name = scene_name +"_section" +line[17]
                            
                        scene_file = CSV + new_scene_name + ".csv"

                        with open(scene_file,"a") as scene_csv:
                            scene_writer = csv.writer(scene_csv)

                            row = []
                            row.append(DATASET) #dataset
                            row.append(new_scene_name) #scene
                            row.append(int(line[1])) # frame
                            row.append(int(line[0])) # id
                            row.append(float(line[4]) * feet_meters) #x
                            row.append(float(line[5]) * feet_meters) #y

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