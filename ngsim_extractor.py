import csv
import os
import time
import extractors.helpers as helpers
import numpy as np
from scipy.spatial.distance import euclidean
import math

DATASETS = "./data/datasets/"
DATASET = "ngsim"
DATAFILE = DATASETS + DATASET + "/" + DATASET + ".csv"
CSV = "./data/csv/"
MAIN = "main/"

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
    scene_names = ["lankershim" ]



    csv_files = helpers.get_dir_names(CSV)
    for csv_ in csv_files:
        for scene_name in scene_names:
            if scene_name in csv_:
                scene_file = CSV + csv_
                # print(scene_file)
                if os.path.exists(scene_file):
                    os.remove(scene_file)
    print("Done!")

    # scene_names = ["lankershim"]
    # print("extracting subscenes id")
    # subscenes = get_subscenes(scene_names)
    # print("Done!")
    # print(time.time()-start)

    print("processing extraction...")

    feet_meters = 0.3048
    for scene_name in scene_names:
        print("processing scene: " + scene_name)
        with open(DATAFILE) as csv_reader:
            csv_reader = csv.reader(csv_reader, delimiter=',')
            

            dataset_file = CSV + MAIN + DATASET + ".csv"
            with open(dataset_file,"a") as dataset_csv:
                dataset_writer = csv.writer(dataset_csv)

                line_num = 0
                last_id = -1
                last_pos = [-1,-1]
                for line in csv_reader:

                    if line_num != 0:
                        # line = line.split(",")
                        scene_line = line[-1]
                        

                        if scene_name == scene_line:
                            
                            new_scene_name = scene_name

                            if line[16] != "0" and line[17] == "0":
                                new_scene_name = scene_name +"_inter" +line[16]  ####################
                                


                            # elif line[17] != "0" and line[16] == "0":
                            #     new_scene_name = scene_name +"_section" +line[17]
                        
                                scene_file = CSV + new_scene_name + ".csv"

                                if new_scene_name != scene_name:

                                    new_id = int(line[0])
                                        
                                    new_pos = [
                                        float(line[4]) * feet_meters,
                                        float(line[5]) * feet_meters
                                    ]

                                                             

                                    if  last_id == new_id:
                                        t = [0.,0.]
                                        b = [0.,0.]
                                        disp = np.subtract(new_pos,last_pos)
                                        norm = np.linalg.norm(disp)
                                        if norm == 0.:
                                            norm = 1
                                        disp /= norm
                                        axis = [1,0]

                                        theta = np.arccos(np.dot(axis,disp))

                                        disp1 = np.subtract(new_pos,first_pos)
                                        norm1 = np.linalg.norm(disp1)
                                        if norm1 == 0.:
                                            norm1 = 1
                                        disp1 /= norm1

                                        theta1 = np.arccos(np.dot(axis,disp1))


                                        # if new_pos[0] < 0:
                                        #     print(theta)  
                                        length = float(line[8]) * feet_meters
                                        width = float(line[9]) * feet_meters
                                        # if math.pi/4. < theta and theta < 3.*math.pi / 4.:

                                        
                                    
                                        # t,b,new_pos = helpers.get_bbox(theta,theta1,width,length,new_pos[0],new_pos[1])
                                        t =[-1,-1]
                                        b =[-1,-1]

                                        
                                        with open(scene_file,"a") as scene_csv:
                                            subscene_writer = csv.writer(scene_csv)

                                            row = []
                                            row.append(DATASET) #dataset
                                            row.append(new_scene_name) #scene
                                            row.append(int(line[1])) # frame
                                            row.append(new_id) # id
                                            row.append(new_pos[0]) #x
                                            row.append(new_pos[1])  #y

                                            row.append(t[0]) #xl
                                            row.append(t[1]) #yl
                                            row.append(b[0]) #xb
                                            row.append(b[1]) #yb

                                            row.append(dict_type[int(line[10])]) # type

                                            subscene_writer.writerow(row)
                                            dataset_writer.writerow(row)
                                    else:
                                        first_pos = new_pos
                                    
                                    last_pos = new_pos                                    
                                    last_id = new_id

                    line_num += 1
    print(time.time()-start)
    



if __name__ == "__main__":
    main()