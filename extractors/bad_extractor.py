import os
import helpers
from helpers import bb_intersection_over_union as iou
import csv
import pandas as pd
import time
import itertools
import numpy as np
from helpers import remove_char,parse_init_line,parse_boxes_line

ROOT = "./datasets/"
DATASET = "bad"
DATA = ROOT + DATASET + "/"
BOXES_PATH = DATA + "boxes.csv"
INIT_PATH = DATA + "init.csv"
TRAJECTORIES_PATH = DATA + "trajectories.csv"
dict_classes = {'7': 'car', '15':'pedestrian', '6': 'car', '14': 'pedestrian'}
CSV_PATH = "./csv/"

NB_FRAME = 1000


PRINT_EVERY = 300



from skimage.transform import ProjectiveTransform

HOMOGRAPHY = "datasets/bad/homography/"
# def compute_homography_data(real_corners,cam_corners,save = True):
#     real = np.loadtxt(real_corners)
#     cam = np.loadtxt(cam_corners)
#     transformer = ProjectiveTransform()
#     transformer.estimate(real,cam)
#     homography = transformer.params
#     if not save:    
#         return homography
#     np.savetxt(HOMOGRAPHY + "homography.txt",homography)
#     return

def apply_homography_box(box,homography):
    new_box = []
    transformer = ProjectiveTransform(matrix = homography)
    for p in transformer.inverse(box[:2])[0]:
        new_box.append(p)
    for p in transformer.inverse(box[2:])[0]:
        new_box.append(p)
    return new_box

def main():
    s = time.time()

    bad_csv = CSV_PATH + DATASET + ".csv"
    if os.path.exists(bad_csv):
        os.remove(bad_csv)
  
    homography =  np.loadtxt(HOMOGRAPHY + "homography.txt")
    transformer = ProjectiveTransform(matrix = homography)

    COUNTER = 0
    print("Computing frame to line")
    frame_to_line = {}
    
    line_count = 0

    available_boxes = {}

    with open(BOXES_PATH) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if line_count != 0:
                bbox,bframe = parse_boxes_line(row)
                available = [True for _ in bbox]
                available_boxes[bframe] = available

                if bframe in frame_to_line:
                    frame_to_line[bframe] = min(line_count,frame_to_line[bframe])
                else:
                    frame_to_line[bframe] = line_count
            line_count +=1

    with open(bad_csv,"a") as csv_file:
        writer = csv.writer(csv_file)

        with open(INIT_PATH) as csv_file:
            init_reader = csv.reader(csv_file, delimiter=',')
            line_count_init = 0
            for row in init_reader:
                if line_count_init != 0:
                    trajectory_id,class_id,box,frame = parse_init_line(row)

                    ######################
                    # print(box)
                    box = apply_homography_box(box,homography)
                    # print(box)
                    #######################


                    if trajectory_id%PRINT_EVERY == 0:
                        print("trajectory n°"+str(trajectory_id))
                        print(time.time() - s)

                    last_frame_added = -1
                    last_row_added = []
                    with open(BOXES_PATH,'r') as csv_reader:
                        current_line = 0
                        boxes_reader = itertools.islice(csv.reader(csv_reader, delimiter=','),frame_to_line[frame],None)
                    
                        first_row = next(boxes_reader)
                        current_line += 1
                        bboxes,bframe = parse_boxes_line(first_row)
                        
                        
                        
                        while bframe < frame + NB_FRAME :
                            if last_frame_added < bframe and bframe + 1 > frame: 
                                # print(bboxes,available_boxes[bframe])
                                k = 0
                                for bbox,is_available in zip(bboxes,available_boxes[bframe]):

                                    ######################
                                    # print(bbox)
                                    bbox = apply_homography_box(bbox,homography)
                                    # print(bbox)
                                    #######################

                                    if is_available:
                                        o = iou(box,bbox)
                                        if  o > 0.9 :

                                            x = (bbox[0] + bbox[2])/2.0
                                            y = (bbox[1] + bbox[3])/2.0 # middle of bounding box
                                            # y = bbox[3] # paper version
                                            point = [x,y]
                                            top_left = [bbox[0],bbox[1]]
                                            bottom_down = [bbox[2],bbox[3]]
                                            
                                            # point = transformer.inverse(point)[0]
                                            

                                            # top_left = transformer.inverse(top_left)[0]
                                            # bottom_down = transformer.inverse(bottom_down)[0]




                                            new_line = []
                                            new_line.append(DATASET) # dataset label
                                            new_line.append(DATASET) # subscene label
                                            new_line.append(bframe) #frame
                                            new_line.append(trajectory_id) #id
                                            new_line.append(point[0]) #x
                                            new_line.append(point[1]) #y
                                            new_line.append(top_left[0])# xmin. The top left x-coordinate of the bounding box.
                                            new_line.append(top_left[1])# ymin The top left y-coordinate of the bounding box.
                                            new_line.append(bottom_down[0])# xmax. The bottom right x-coordinate of the bounding box
                                            new_line.append(bottom_down[1])# ymax. The bottom right y-coordinate of the bounding box
                                            new_line.append(dict_classes[class_id]) # label type of agent   


                                            if last_row_added != [] and last_frame_added + 1 < bframe:
                                                for i in range(last_frame_added + 1, bframe):
                                                    last_row_added[2] = i
                                                    writer.writerow(last_row_added)

                                            writer.writerow(new_line)
                                            COUNTER += 1
                                            available_boxes[bframe][k] = False

                                            box = bbox
                                            last_frame_added = bframe
                                            last_row_added = new_line

                                            

                                            break
                                    k += 1
                            try:
                                brow = next(boxes_reader)
                                
                                bboxes,bframe = parse_boxes_line(brow)
                                
                            except:
                                break
                            current_line += 1
                        
                line_count_init += 1  
    print("Done!")
    print(time.time() - s)

    print( "number of used boxes: " + str(COUNTER))

    count_true = len([1 for b in available_boxes for t in available_boxes[b] if t == True])
    count_false = len([1 for b in available_boxes for t in available_boxes[b] if t == False])
    count_total = len([1 for b in available_boxes for t in available_boxes[b] ])

    print(count_true,count_false,count_total)

if __name__ == "__main__":
    main()