import os
import numpy as np
from scipy.spatial.distance import euclidean

"""
    Get the directories contained in a directory
    path: directory path
    lower: set the names to lower case
    ordered: order directory names lexygraphically
    descending: descending order for directory names
"""
def get_dir_names(path,lower = True,ordered = True,descending = False):
    dir_names = []
    dirs = os.listdir(path)
    if ordered:
        dirs = sorted(dirs,key = str, reverse = descending)
    for x in dirs:
        if lower:
            x = x.lower()
        dir_names.append(x)
    return dir_names


"""
    Parse and add scene specific informations to a line
    line : a line of the original SDD dataset
    scene: scene name + subscene number
    dataset: name of the dataset
"""
def parse_line(line,scene, dataset ):
    line = line.split(" ")

    new_line = []    
    
    xa = float(line[1])
    ya = float(line[2])
    xb = float(line[3])
    yb = float(line[4])

    x = str((xa + xb)/2)
    y = str((ya+yb)/2 )

    new_line.append(dataset) # dataset label
    new_line.append(scene)   # subscene label
    new_line.append(line[5]) #frame
    new_line.append(line[0]) #id

    new_line.append(x) #x
    new_line.append(y) #y
    new_line.append(line[1]) # xmin. The top left x-coordinate of the bounding box.
    new_line.append(line[2]) # ymin The top left y-coordinate of the bounding box.
    new_line.append(line[3]) # xmax. The bottom right x-coordinate of the bounding box.
    new_line.append(line[4]) # ymax. The bottom right y-coordinate of the bounding box.

    new_line.append(line[9]) # label type of agent    

    return new_line

"""
    Intersection over Union between two bounding boxes
    box = [xtopleft,ytopleft,xbottomright,ybottomright]
""" 

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


"""
    Remove a list of chars from string
"""
def remove_char(chars,string):
    # chars_to_remove = ['{','}',' ','\'']
    sc = set(chars)
    string = ''.join([c for c in string if c not in sc])
    return string

"""
    In bad extractor parse row from text file obtained
    after parsing the init cpkl file
"""
def parse_init_line(row):
    trajectory_id = int(row[0])
    class_id = row[1]
    box = remove_char(['[',']'],row[2]).split(",")
    box = [float(b) for b in box]
    frame = int(row[3])
    return trajectory_id,class_id,box,frame

"""
    In bad extractor parse row from text file obtained
    after parsing the boxes cpkl file
"""

def parse_boxes_line(row):

    nb_points = 4 
    
    box = remove_char(['[',']'],row[0]).split(",")
    box = [float(b) for b in box]
    boxes = []
    for j in range(int(len(box)/nb_points)):
        sub_box = []
        for i in range(nb_points):
            sub_box.append(float(box[nb_points * j + i]))
        boxes.append(sub_box)

    frame = int(row[1])
    return boxes,frame



"""
    Get the first timestep for a scene in fsc dataset
"""
def frame_corrector(subscene_path):
    with open(subscene_path,"r") as a:

        start_frame = None
        # last_frame = 0.

        # min_step = 10000
        for k,line in enumerate(a):

            frame = float(line.split(" ")[0])
            if k == 0:
                start_frame = frame
            # step = frame - last_frame
            # if step < min_step:
            #     min_step = step
            
            # last_frame = frame
        # min_step = round(min_step,5)
        # gap = round(last_frame-start_frame,5)
        # nb_frame = int(gap/min_step)
    return start_frame


"""
    Parse and add scene specific informations to a line
    line : a line of the original fsc dataset
    scene: scene name + subscene number
    dataset: name of the dataset
    min_step: framrate * 100
    start_frame: first timestamp of the scene
"""   

def parse_line_fsc(line,scene_name, dataset_name,min_step,start_frame):

    undefined = -1

    line = line.split(" ")

    timestamp = round(float(line[0]),2)*100
    start_frame = round(start_frame,2)*100
    dt = timestamp - start_frame

    frame = int(dt/min_step)
    id_ = line[1]
    
    r = -1
    theta = -1
    x = -1
    y = -1

    if "" != line[3] and "" != line[-1]:
        
        r = float(line[3])     
        theta = float(line[-1])

        x = r * np.cos(theta)
        y = r * np.sin(theta)

    new_line = []

    new_line.append(dataset_name) # dataset label
    new_line.append(scene_name)   # subscene label
    new_line.append(frame) #frame
    new_line.append(id_) #id

    new_line.append(x) #x
    new_line.append(y) #y
    new_line.append(undefined) # xmin. The top left x-coordinate of the bounding box.
    new_line.append(undefined) # ymin The top left y-coordinate of the bounding box.
    new_line.append(undefined) # xmax. The bottom right x-coordinate of the bounding box.
    new_line.append(undefined) # ymax. The bottom right y-coordinate of the bounding box.

    new_line.append(undefined) # label type of agent    

    return new_line

def transpose(theta,points,x,y):
    R = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ]
    T = [x,y]
    new_points = []

    for point in points:
        new_point = np.add(np.matmul(R,point), T)
        new_points.append(new_point)
    return np.array(new_points)



def get_bbox(theta,width,length,x,y):
    t,b = [],[]

    points = [
        [-length,width/2.0],
        [-length,-width/2.0],
        [0,width/2.0],
        [0,-width/2.0], 
        [-length/2.0,0],         
    ]

    new_points = transpose(theta,points,x,y)
    # new_points = transpose(theta,points,0,0)

    # dists = [ euclidean(p,[0,0]) for p in points]

    # i_min = np.argmin(dists)
    # i_max = np.argmax(dists)

    t = []
    b = []
    pm = new_points[-1]
    for p in new_points[:-1]:
        if p[0] < pm[0] and p[1] > pm[1]:
            t = p
        if p[0] > pm[0] and p[1] < pm[1]:
            b = p
    
    

    # return new_points
    return t,b,pm