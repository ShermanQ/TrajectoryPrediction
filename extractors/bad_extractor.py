import os
import helpers
from helpers import bb_intersection_over_union as iou
import csv
import pandas as pd
import time
import itertools
import numpy as np

ROOT = "./datasets/"
DATASET = "bad"
DATA = ROOT + DATASET + "/"
BOXES_PATH = DATA + "boxes.csv"
INIT_PATH = DATA + "init.csv"
TRAJECTORIES_PATH = DATA + "trajectories.csv"
dict_classes = {'7': 'car', '15':'pedestrian', '6': 'car', '14': 'pedestrian'}
CSV_PATH = "./csv/"

NB_FRAME = 300


PRINT_EVERY = 300

def remove_char(chars,string):
    # chars_to_remove = ['{','}',' ','\'']
    sc = set(chars)
    string = ''.join([c for c in string if c not in sc])
    return string

def parse_init_line(row):
    trajectory_id = int(row[0])
    class_id = row[1]
    box = remove_char(['[',']'],row[2]).split(",")
    box = [float(b) for b in box]
    frame = int(row[3])
    return trajectory_id,class_id,box,frame

def parse_boxes_line(row):
    
    box = box = remove_char(['[',']'],row[0]).split(",")
    box = [float(b) for b in box]
    frame = int(row[1])
    return box,frame

def main():
    # df = pd.read_csv(BOXES_PATH)

    # for row in df.iterrows():
    #     pass
    s = time.time()

    bad_csv = CSV_PATH + DATASET + ".csv"
    if os.path.exists(bad_csv):
        os.remove(bad_csv)
    
    # nb_lines = 0
    # with open(BOXES_PATH) as f:
    #     nb_lines = sum(1 for line in f)
        

    print("Computing frame to line")
    frame_to_line = {}
    
    line_count = 0
    with open(BOXES_PATH) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if line_count != 0:
                bbox,bframe = parse_boxes_line(row)
                if bframe in frame_to_line:
                    frame_to_line[bframe] = min(line_count,frame_to_line[bframe])
                else:
                    frame_to_line[bframe] = line_count
            line_count +=1


    available_boxes = np.ones(line_count)
    



    with open(bad_csv,"a") as csv_file:
        writer = csv.writer(csv_file)

        with open(INIT_PATH) as csv_file:
            init_reader = csv.reader(csv_file, delimiter=',')
            line_count_init = 0
            for row in init_reader:
                # print(row)
                if line_count_init != 0:
                    trajectory_id,class_id,box,frame = parse_init_line(row)

                    if trajectory_id%PRINT_EVERY == 0:
                        print("trajectory n°"+str(trajectory_id))
                        print(time.time() - s)
                

                    with open(BOXES_PATH,'r') as csv_reader:
                        current_line = 0
                        # print(frame, frame + NB_FRAME)
                        # print(frame)
                        boxes_reader = itertools.islice(csv.reader(csv_reader, delimiter=','),frame_to_line[frame],None)
                       
                        
                        if current_line == 0:
                            _ = next(boxes_reader)
                            

                        # next(itertools.islice(csv.reader(f), N, None)
                        first_row = next(boxes_reader)
                        current_line += 1
                        bbox,bframe = parse_boxes_line(first_row)
                        
                        
                        while bframe < frame + NB_FRAME:
                            # print(bframe)
                            if available_boxes[current_line] and bframe + 1 > frame: 
                                o = iou(box,bbox)
                                # print(box)
                                # print(bbox)
                                # print(o)
                                if  o > 0.9 :
                                    available_boxes[current_line] = 0
                                    new_line = []
                                    new_line.append(DATASET) # dataset label
                                    new_line.append(DATASET) # subscene label
                                    new_line.append(bframe) #frame
                                    new_line.append(trajectory_id) #id
                                    new_line.append(0) #x
                                    new_line.append(0) #y
                                    new_line.append(bbox[0])# xmin. The top left x-coordinate of the bounding box.
                                    new_line.append(bbox[1])# ymin The top left y-coordinate of the bounding box.
                                    new_line.append(bbox[2])# xmax. The bottom right x-coordinate of the bounding box
                                    new_line.append(bbox[3])# ymax. The bottom right y-coordinate of the bounding box
                                    new_line.append(dict_classes[class_id]) # label type of agent   

                                    writer.writerow(new_line)

                                    box = bbox
                            # print("trajectory n°"+str(trajectory_id))
                            try:
                                brow = next(boxes_reader)
                                bbox,bframe = parse_boxes_line(brow)
                            except:
                                break
                            current_line += 1
                            # print(brow)
                            
                            

                # if line_count_init != 0:
                #     break
                line_count_init += 1  
    print("Done!")
    print(time.time() - s)

    print( "number of not used lines: " + str(sum(available_boxes))
    nb_box_per_traj = np.zeros(line_count_init)

    with open(bad_csv) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        line_ = 0
        for row in reader:
            # print(row)
            if line_ != 0:
                nb_box_per_traj[int(row[3])] += 1
            
            line_ += 1
    print("mean number of boxes per trajectorie: " + str(np.mean(nb_box_per_traj)), "std: " + str(np.std(nb_box_per_traj)))




# Note generate frame when stopped

if __name__ == "__main__":
    main()