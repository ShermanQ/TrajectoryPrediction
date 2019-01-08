import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories

import time


import helpers_a
import json
import csv

FPS_PATH = "./framerates.json"

ROOT = "./../"
CSV = ROOT + "extractors/csv/"





    # print(trajectory["id"],np.mean(deltas),np.std(deltas))
        
def write_frame(frame,file_name,new_frame_id):

    with open(file_name,"a") as file_:
        file_writer = csv.writer(file_)
        for key in frame:
            if key != "frame":
                new_line = []
                new_line.append(frame[key]["dataset"])
                new_line.append(frame[key]["scene"])
                new_line.append(new_frame_id)
                new_line.append(key)
                for e in frame[key]["coordinates"]:
                    new_line.append(e)

                for e in frame[key]["bbox"]:
                    new_line.append(e)

                new_line.append(frame[key]["type"])

                file_writer.writerow(new_line)
    


def main():
    dir_list = ["main","new_rates"]
   
    csvs = [ f for f in get_dir_names(CSV,lower = False) if f not in dir_list]


    csv_file = csvs[1]
    file_path = CSV + csv_file

    print(file_path)
    json_file = open(FPS_PATH)
    framerates = json.load(json_file)
    temp_path = "./temp.txt"


    name = csv_file.split(".")[0]
    former_rate = float(framerates[name])
    new_rate = 2.5


    destination_file = CSV + "new_rates/"+ name + "_" +  str(former_rate)+"to" + str(new_rate) +".csv"


    if os.path.exists(destination_file):
        os.remove(destination_file)

    print(destination_file)
    rate_ratio = int(former_rate/new_rate)
    print(rate_ratio)

    extract_frames(file_path,temp_path,save=True)
    with open(temp_path) as frames:
        counter = 0
        for i,frame in enumerate(frames):
            frame = json.loads(frame)
            if i % rate_ratio == 0:
                write_frame(frame,destination_file,counter)
                counter += 1
    
    os.remove(temp_path)


    # csv = CSV + "fsc_0.csv"
    # for csv in csvs:
    #     print(csv)
    # deltas,lengths,outliers,props,missing_segments,nb_missing = helpers_a.trajectories_continuity(csv,temp_path = "./temp.txt")

    # print(missing_segments)
    # for key in nb_missing:
    #     print(key,nb_missing[key],lengths[key])

    # print(outliers)
    # for csv in csvs:
    #     print(csv)
    # conflicts,nb_collisions_total,nb_objects_total,ious_total,ious_conflict_total = helpers_a.collisions_in_scene(csv) 
    #     print(time.time() - s)               
    # print(time.time() - s)
    # missing_values_multiple(csvs)




if __name__ == "__main__":
    main()