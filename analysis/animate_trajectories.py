import cv2
import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extractors.helpers as helpers
import helpers_v as vis

import csv
import json
import matplotlib.cm as cm
from collections import deque

ROOT = "./../"
CSV = ROOT + "extractors/csv/"



def main():

    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    file_path = CSV + "deathCircle1.csv"
    framerate = 30
    factor_div = 2.0
    nb_last_steps = 200
    new_color_index = 0
    color_dict = {}
    temp_path = "./temp.txt"

    helpers.extract_frames(file_path,temp_path,save = True)

    try:
        min_,max_ = vis.find_bounding_coordinates(temp_path)

        
        w,h = vis.get_scene_image_size(min_,max_,factor_div = factor_div)
        
        
        


        # Create a black image
        img = np.zeros((h,w,3), np.uint8)

        last_frames = deque([])

        with open(temp_path) as frames:
            for frame in frames:
                frame = json.loads(frame)

                if len(last_frames) == nb_last_steps:
                    last_frames.popleft()
                last_frames.append(frame)

                


                img1 = img.copy()


                img1,color_dict,new_color_index = vis.plot_current_frame(frame,img1,color_dict,new_color_index,factor_div)

                img1 = vis.plot_last_steps(img1,frame,last_frames,color_dict,factor_div=factor_div)
     

                img1 = vis.plot_frame_number(w,h, img1,frame)
                

                cv2.imshow('image1',img1)
                cv2.waitKey(int(1000/framerate))

        cv2.destroyAllWindows()
        os.remove(temp_path)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        os.remove(temp_path)



    

if __name__ == "__main__":
    main()
