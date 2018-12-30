import os
import helpers
from helpers import bb_intersection_over_union as iou
import csv
import pandas as pd
import time
import itertools
import numpy as np
import helpers

ROOT = "./datasets/"
DATASET = "fsc"
DATA = ROOT + DATASET + "/"
RADAR_SUFFIX = "radar_data.txt"
SAMPLING_RATE = 4 #0.04 * 100
CSV_PATH = "./csv/"
MAIN = "main/"




def main():
    print("Starting extraction")
    start = time.time()

    main_scene_names = helpers.get_dir_names(DATA, lower = False)

    fsc_csv = CSV_PATH + MAIN + DATASET + ".csv"

    if os.path.exists(fsc_csv):
        os.remove(fsc_csv)

    for i,scene in enumerate(main_scene_names):   

        

        with open(fsc_csv,"a") as csv_file:
            writer = csv.writer(csv_file)

            

            scene_csv = CSV_PATH + DATASET + "_" + str(i)+".csv"

            if os.path.exists(scene_csv):
                os.remove(scene_csv)

            with open(scene_csv,"a") as csv_scene:
                writer_scene = csv.writer(csv_scene)
                
                print("Processing scene: " + scene)

                subscene_path = DATA + scene + "/" + RADAR_SUFFIX
                        

                start_frame = helpers.frame_corrector(subscene_path)


                with open(subscene_path,"r") as a:
                    for line in a:
                        new_line = helpers.parse_line_fsc(line,DATASET + "_" + str(i), DATASET ,SAMPLING_RATE,start_frame)
                        writer.writerow(new_line)
                        writer_scene.writerow(new_line)
        
                    
    print("Execution time: " + str(time.time() - start) + " s") 



if __name__ == "__main__":
    main()
