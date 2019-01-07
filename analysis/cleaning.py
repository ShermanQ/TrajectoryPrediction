import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories

import time
import json
import numpy as np
from scipy.spatial.distance import euclidean

import helpers_a

ROOT = "./../"
CSV = ROOT + "extractors/csv/"





    # print(trajectory["id"],np.mean(deltas),np.std(deltas))
        


def main():
    csvs = [ CSV + f for f in get_dir_names(CSV,lower = False) if f != "main"]
    # s = time.time()   

    csv = csvs[1]
    # csv = CSV + "fsc_0.csv"
    # for csv in csvs:
    #     print(csv)
    deltas,lengths,outliers,props,missing_segments,nb_missing = helpers_a.trajectories_continuity(csv,temp_path = "./temp.txt")

    # print(missing_segments)
    # for key in nb_missing:
    #     print(key,nb_missing[key],lengths[key])

    # print(outliers)
    # for csv in csvs:
    #     print(csv)
    # nb_collisions_total,nb_objects_total,ious_total,ious_conflict_total = helpers_a.collisions_in_scene(csv) 
    #     print(time.time() - s)               
    # print(time.time() - s)
    # missing_values_multiple(csvs)




if __name__ == "__main__":
    main()