import sys 
import os
# from extractors.helpers import get_dir_names,extract_frames
import extractors.helpers as helpers
import time


import analysis.helpers_a
import json
import csv

FPS_PATH = "./parameters/framerates.json"

ROOT = "./"
CSV = ROOT + "data/csv/"





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

                new_line.append(frame[key]["type"].split("\n")[0])


                file_writer.writerow(new_line)
def framerate_manager(framerates_json,csv_file,file_path,destination_root,new_rate = 2.5,temp_path = "./data/temp/temp.txt"):   

    
    json_file = open(framerates_json)
    framerates = json.load(json_file)
    
    name = csv_file.split(".")[0]
    print(csv_file)
    former_rate = float(framerates[name])
    
    rate_ratio = int(former_rate/new_rate)

    destination_file = destination_root + name + "_" +  str(former_rate)+"to" + str(new_rate) +".csv"
    if os.path.exists(destination_file):
        os.remove(destination_file)

    

    helpers.extract_frames(file_path,temp_path,save=True)
    with open(temp_path) as frames:
        counter = 0
        for frame in frames:
            frame = json.loads(frame)
            i = frame["frame"]
            # frame number 
            if i % rate_ratio == 0:
                write_frame(frame,destination_file,counter)
                counter += 1    
    os.remove(temp_path)

def main():
    dir_list = ["main","new_rates"]
    dir_list = ["lankershim_inter2.csv"]
   
    # csvs = [ f for f in helpers.get_dir_names(CSV,lower = False) if f not in dir_list]
    csvs = [ f for f in helpers.get_dir_names(CSV,lower = False) if f in dir_list]



    # csv_file = csvs[1]
    # file_path = CSV + csv_file
    # destination_root = CSV + "new_rates/"
    # framerate_manager(FPS_PATH,csv_file,file_path,destination_root,temp_path = "./temp.txt") 

    # print(file_path)
    


    destination_root = CSV + "new_rates/"

    s = time.time()

    for csv_file in csvs:
        print(csv_file)
        file_path = CSV + csv_file

        framerate_manager(FPS_PATH,csv_file,file_path,destination_root,new_rate = 1.0,temp_path = "./data/temp/temp.txt")  
        print(time.time() - s)
    print(time.time() - s)

  
    # csv_file = "koper.csv"
    # print(csv_file)
    # file_path = CSV + csv_file

    # framerate_manager(FPS_PATH,csv_file,file_path,destination_root,temp_path = "./temp.txt")  
    # print(time.time() - s)


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