import os
import helpers
from helpers import bb_intersection_over_union as iou
import csv
import pandas as pd
import time
import itertools

ROOT = "./datasets/"
DATASET = "bad"
DATA = ROOT + DATASET + "/"
BOXES_PATH = DATA + "boxes.csv"
INIT_PATH = DATA + "init.csv"
TRAJECTORIES_PATH = DATA + "trajectories.csv"
dict_classes = {'7': 'car', '15':'pedestrian', '6': 'unknown'}
CSV_PATH = "./csv/"

NB_FRAME = 300


PRINT_EVERY = 50

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
                

                    with open(BOXES_PATH) as csv_file:
                        # print(frame, frame + NB_FRAME)
                        # print(frame)
                        boxes_reader = itertools.islice(csv.reader(csv_file, delimiter=','),frame,None)

                        if frame == 0:
                            _ = next(boxes_reader)

                        # next(itertools.islice(csv.reader(f), N, None)
                        first_row = next(boxes_reader)
                        bbox,bframe = parse_boxes_line(first_row)
                        
                        
                        while bframe < frame + NB_FRAME:
                            # print(bframe)
                            if bframe + 1 > frame: 
                                o = iou(box,bbox)
                                # print(box)
                                # print(bbox)
                                # print(o)
                                if  o > 0.9 :
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
                            brow = next(boxes_reader)
                            # print(brow)
                            
                            bbox,bframe = parse_boxes_line(brow)

                # if line_count_init != 0:
                #     break
                line_count_init += 1  
    print("Done!")
    print(time.time() - s)
                

# Note generate frame when stopped

if __name__ == "__main__":
    main()