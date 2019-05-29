import cv2
import numpy as np 

import sys 
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import helpers
import analysis.helpers_v as vis

import csv
import json
import matplotlib.cm as cm
from collections import deque



# CSV = "./data/csv/"

# CSV = "./data/csv/new_rates/"

# python animate_trajectories.py parameters/data.json scene 1
def main():

    # file_path = CSV + "new_rates/gates8_30.0to2.5.csv"
    # file_path = CSV + "bad.csv"
    # file_path = CSV + "lankershim_inter2_10.0to1.0.csv"
    # file_path = CSV + "deathCircle1.csv"
    # file_path = CSV + "01_tracks.csv"

    args = sys.argv
    data = json.load(open(args[1]))
    original = int(args[3])

    file_path = ""
    if original:
        file_path = data["extracted_datasets"] + args[2] + ".csv"
    else:
        file_path = data["preprocessed_datasets"] + args[2] + ".csv"
    

    

    framerate = 2.5
    # factor_div = 3/2
    plot_last_step = True
    nb_last_steps = 200
    new_color_index = 0
    color_dict = {}
    temp_path = "./data/temp/temp.txt"
    target_size = 1000
    margin = 10

    offset = [0,0]

    helpers.extract_frames(file_path,temp_path,save = True)

    try:
        min_,max_ = vis.find_bounding_coordinates(temp_path)
        w,h = vis.get_scene_image_size(min_,max_)

        print(min_,max_)
        print(w,h)
        factor_div  = np.max([w,h]) / target_size

        w,h = int(w/factor_div) + margin,int(h/factor_div)+margin

        offset = np.divide(min_,-factor_div)

        print(factor_div)
        print(offset)

        print(w,h)
        


        # Create a black image
        img = np.zeros((h,w,3), np.uint8)

        last_frames = deque([])

        with open(temp_path) as frames:
            for frame in frames:
                frame = json.loads(frame)
                # print(frame)
                if len(last_frames) == nb_last_steps:
                    last_frames.popleft()
                last_frames.append(frame)

                


                img1 = img.copy()


                img1,color_dict,new_color_index = vis.plot_current_frame(frame,img1,color_dict,new_color_index,factor_div,offset, display_box= True)

                if plot_last_step:
                    img1 = vis.plot_last_steps(img1,frame,last_frames,color_dict,factor_div=factor_div,offset = offset)
     

                img1 = vis.plot_frame_number(w,h, img1,frame)
                
                # if file_path == CSV + "bad.csv":
                img1 = vis.plot_scene_name(w,h, img1,frame)




                cv2.imshow('image1',img1)
                cv2.waitKey(int(1000/framerate))

        cv2.destroyAllWindows()
        os.remove(temp_path)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        os.remove(temp_path)



    

if __name__ == "__main__":
    main()
