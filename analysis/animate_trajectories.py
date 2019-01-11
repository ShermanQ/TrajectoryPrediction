import cv2
import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extractors.helpers as helpers

import csv
import json
import matplotlib.cm as cm

ROOT = "./../"
CSV = ROOT + "extractors/csv/"

def find_bounding_coordinates(frames_path):
    minx = 10e30
    miny = 10e30
    maxx = 10e-30
    maxy = 10e-30

    with open(frames_path) as frames:
        for frame in frames:
            frame = json.loads(frame)
            for id_ in frame:
                if id_ != "frame":
                    coordinates = frame[id_]["coordinates"]

                    x = coordinates[0]
                    y = coordinates[1]

                    if x > maxx:
                        maxx = x
                    if y > maxy:
                        maxy = y
                    if x < minx:
                        minx = x
                    if y < miny:
                        miny = y
    return [minx,miny],[maxx,maxy]

def get_scene_image_size(min_,max_,factor_div = 2.0, E = 100):
    fx = max_[0] + E
    fy = max_[1] + E
    if min_[0] < 0:
        fx -= min_[0]
    if min_[1] < 0:
        fy -= min_[1]
    return int(fx/factor_div),int(fy/factor_div)
def get_number_object(file_path):
    dict_ = {}
    with open(file_path) as scene:
        for line in scene:
            line = line.split(",")
            id_ = line[3]

            dict_[id_] = 1
    return len(dict_.keys())


def main():

    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    file_path = CSV + "deathCircle1.csv"
    framerate = 30

    temp_path = "./temp.txt"
    helpers.extract_frames(file_path,temp_path,save = True)

    try:
        min_,max_ = find_bounding_coordinates(temp_path)

        factor_div = 2.0
        w,h = get_scene_image_size(min_,max_,factor_div = factor_div)
        
        
        new_color_index = 0
        color_dict = {}


        # Create a black image
        img = np.zeros((w,h,3), np.uint8)

        with open(temp_path) as frames:
            for frame in frames:
                frame = json.loads(frame)

                img1 = img.copy()

                for id_ in frame:
                    if id_ != "frame":

                        if id_ not in color_dict:
                            # color_dict[id_] = tuple(colors[new_color_index][:3] * 255)
                            color_dict[id_] = [int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )]

                            
                            new_color_index += 1

                        coordinates = frame[id_]["coordinates"]
                        # print(type(color_dict[id_][0]))
                        
                        
                        cv2.circle(img1,tuple([int(p/factor_div) for p in coordinates]), 5, tuple(color_dict[id_]), -1)
                

                font = cv2.FONT_HERSHEY_SIMPLEX

                
                text_pos = tuple(np.subtract([h,w],[100,25]))
                
                cv2.putText(img1, str(frame["frame"]), text_pos, font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('image1',img1)
                cv2.waitKey(int(1000/framerate))




        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        os.remove(temp_path)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        os.remove(temp_path)


    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img,(0,0),(511,511),(255,0,0),5)

    

if __name__ == "__main__":
    main()
