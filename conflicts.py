import analysis.helpers_a as a 
import sys 
import os
import extractors.helpers as helpers
import numpy as np

import json


names = ["lankershim_inter1_0",        
        "lankershim_inter1_1",        
        "lankershim_inter2_0",
        "lankershim_inter2_1",
        "lankershim_inter3_0",
        "lankershim_inter3_1",
        "lankershim_inter4_0",
        "lankershim_inter4_1",
        "peachtree_inter1_0",
        "peachtree_inter2_0",
        "peachtree_inter3_0",
        "peachtree_inter4_0",
        "peachtree_inter5_0",
        "peachtree_inter1_1",
        "peachtree_inter2_1",
        "peachtree_inter3_1",
        "peachtree_inter4_1",
        "peachtree_inter5_1"]

stats = {}
for name in names:
    path_ = "./data/csv/" + name + ".csv"
    _,nb_conflicts,nb_agents,_,_ = a.collisions_in_scene(path_,temp_path = "./data/temp/temp.txt")
    trajectories_deltas,trajectories_length,trajectories_outliers,_,_,_= a.trajectories_continuity(path_,temp_path = "./data/temp/temp.txt")
    trajectories_outliers = [len(trajectories_outliers[key]) for key in trajectories_outliers]
    stats[name] = {
        "nb_conflicts": str(np.sum(nb_conflicts)),
        "nb_agents": str(np.sum(nb_agents)),
        "nb_outlier_points": str(np.sum([trajectories_outliers[key] for key in trajectories_outliers])),
        "nb_points": str(np.sum([trajectories_length[key] for key in trajectories_length]))
    }
destination = "./data/datasets/conflicts.json"
if os.path.exists(destination):
    os.remove(destination)
with open(destination,"w") as csv_dest:
    json.dump(stats,csv_dest,indent= 3)