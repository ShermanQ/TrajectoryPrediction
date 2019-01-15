import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extractors.helpers as helpers
import helpers_c as cl
import helpers_v as vis

from Extractor import Extractor
from Clusterer import Clusterer



ROOT = "./../"
CSV = ROOT + "extractors/csv/"







def main():

    target_size = 1000
    # file_path = CSV + "new_rates/bad_30.0to2.5.csv"
    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    
    # print(sorted(multistep_clusters[2][0]))

    ##############################
    temp_path = "./temp.txt"
    helpers.extract_frames(file_path,temp_path,save = True)
    min_,max_ = vis.find_bounding_coordinates(temp_path)

    w,h = vis.get_scene_image_size(min_,max_)

    # print(min_,max_)
    # print(w,h)
    factor_div  = np.max([w,h]) / target_size
    w,h = int(w/factor_div),int(h/factor_div)

    offset = np.divide(min_,-factor_div)

    # w,h = vis.get_scene_image_size(min_,max_,factor_div = 2.0)
    img = np.zeros((h,w,3), np.uint8)
    os.remove(temp_path)
    ###############################
    trajectories = helpers.extract_trajectories(file_path)
    

    clusterers = [Clusterer(selected_clusterer= 1),Clusterer(selected_clusterer= 1 )]
    extractors = [Extractor(4),Extractor(3)]

    # clusterers = [Clusterer(selected_clusterer= 1)]
    # extractors = [Extractor(4)]

    multistep_clusters,cluster_hierarchy = cl.multi_step_clustering(trajectories,extractors,clusterers)


    cl.display_clusters(trajectories,multistep_clusters[1],img,offset,factor_div=factor_div, mosaic= True)

    cl.display_parent_children(trajectories,multistep_clusters[1:],cluster_hierarchy[-1],0,img,offset,factor_div)

    cl.display_multi_step_clustering(multistep_clusters,img,trajectories,offset,factor_div)
    
    # ##########################################
    # e = Extractor(selected_extractor = 6,gamma= 1.0)
    # trajectories = [
    #     [[1,1],[2,2],[3,3]],
    #     [[0,1],[0,2],[0,3]],
    #     [[1,4],[5,2],[3,3]],
    #     [[0,0],[2,2],[3,3]],
    # ]

    # d = e.extract_dtw(trajectories)
    # print(d)

    # c = Clusterer(selected_clusterer=3,nb_clusters=2)
    # clusters = c.cluster(d)

    # print(clusters)

if __name__ == "__main__":
    main()