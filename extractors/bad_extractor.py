import os
import helpers
import cPickle

DATASET = "bad"
ROOT = "./datasets/"
DATA = ROOT + DATASET + "/data"
BOXES_SUFFIX = "_bboxes.cpkl"
TRAJECTORIES_SUFFIX = "_trajectories.cpkl"

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


def is_same(box, last_boxes, threshold = 0.8):
    for i,l_box in enumerate(last_boxes):
        if bb_intersection_over_union(box,l_box) >= threshold:
            return i
    return -1


def main():
    current_tracks = []
    list_ = []
    nb_frames = 0
    nb_trajectories = 0
    last_boxes = []

    print("Loading directories names")
    directories = helpers.get_dir_names(DATA,lower = False,ordered = True,descending = False)
    
    for dir_ in directories:
        print("Processing directory: " + dir_)
        bboxes_path = DATA + "/" + dir_ + "/" + dir_ + BOXES_SUFFIX
        trajectories_path = DATA + "/" + dir_ + "/" + dir_ + TRAJECTORIES_SUFFIX

        bboxes = None
        trajectories = None
        with open(bboxes_path,"rb") as bboxes_file:
            bboxes = cPickle.load(bboxes_file)
        with open(trajectories_path, "rb") as trajectories_file:
            trajectories = iter(cPickle.load(trajectories_file))

        new_dir = True   
        for boxes in bboxes:
            traj = []

            if boxes != []:
                traj = next(trajectories)
                

            for i in range(len(traj)):
                
                same = -1
                
                if new_dir and (last_boxes != []):
                    same = is_same(boxes[i], last_boxes)
                    if i != -1 and i != same:
                        temp = current_tracks[same]
                        current_tracks[same] = current_tracks[i]
                        current_tracks[i] = temp

                if i + 1 > len(current_tracks):
                        current_tracks.append([])
                if traj[i]["is_initial_state"] and same == -1:
                    
                    current_tracks[-(1+i)] = nb_trajectories
                    nb_trajectories += 1
                # else:
                    
                line = [DATASET, DATASET, str(nb_frames), str(current_tracks[-(1+i)]), traj[i]['x'], traj[i]['y'] ]
                for e in boxes[i]:
                    line.append(e)
                line.append(traj[i]['cls_label'])                
                list_.append(line)

                
            last_boxes = boxes
            nb_frames += 1
            new_dir = False
            # break
    print("Done!")

    for line in list_:
        print(line)

        
if __name__ == "__main__":
    main()



# import cPickle

# with (open("alberta_cam_original_2017-10-27_09-00-10_trajectories.cpkl", "rb")) as openfile:
#     objects = cPickle.load(openfile)

   

# print(len(objects))
# for o in objects[290:300]:
#     print(o)