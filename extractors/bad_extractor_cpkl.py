import os
import helpers
import cPickle
import csv

DATASET = "bad"
ROOT = "./datasets/"
DATA = ROOT + DATASET + "/data"
BOXES_SUFFIX = "_bboxes.cpkl"
TRAJECTORIES_SUFFIX = "_trajectories.cpkl"
INIT_SUFFIX = "_init_labels.cpkl"
TRAJECTORY_COUNTER = 0
SCENE_LENGTHS = []
CURRENT_SCENE = 0

def get_scene_lengths(directories_path):
    directories = helpers.get_dir_names(directories_path,lower = False,ordered = True,descending = False)

    scene_lengths = [0]
    for i,dir_ in enumerate(directories):
        bboxes_path = DATA + "/" + dir_ + "/" + dir_ + BOXES_SUFFIX
        

        with open(bboxes_path,"rb") as bboxes_file:
            bboxes = cPickle.load(bboxes_file)
            scene_lengths.append(len(bboxes))
    return scene_lengths

def line_boxes(box,fieldnames,counter):
    line = [box.tolist()]
    line.append(counter)

    new_line = {}
    for i,key in enumerate(fieldnames):
        new_line[key] = line[i]
    return new_line

def line_init(init,fieldnames,counter):
    global TRAJECTORY_COUNTER
    line = [TRAJECTORY_COUNTER]
    for key in init:
        if key == "bbox":
            line.append(init[key].tolist())
        elif key != "trajectory_label":
            line.append(init[key])
    line.append(counter)

    new_line = {}
    for i,key in enumerate(fieldnames):
        new_line[key] = line[i]

    TRAJECTORY_COUNTER += 1
    return new_line

def line_trajectory(trajectory,fieldnames,counter):
    global SCENE_LENGTHS
    global CURRENT_SCENE
    # print(trajectory)

    if SCENE_LENGTHS == []:
        # print("in")
        return
    line = []
    for key in trajectory:
        if key != "t":
            line.append(trajectory[key])

    t = str(int(trajectory["t"]) + SCENE_LENGTHS[CURRENT_SCENE] )

    line.append(t)
    # print(line)

    new_line = {}
    for i,key in enumerate(fieldnames):
        new_line[key] = line[i]


    return new_line



def cpkl_to_csv_alt(save_path,directories,fieldnames,suffix,line_function):

    global CURRENT_SCENE
    CURRENT_SCENE = 0
    counter = 0
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path,"a") as csv_scene:
        # writer_scene = csv.writer(csv_scene)
        
        writer = csv.DictWriter(csv_scene, fieldnames=fieldnames)
        writer.writeheader()

        for i,dir_ in enumerate(directories):
            bboxes_path = DATA + "/" + dir_ + "/" + dir_ + suffix
                

            with open(bboxes_path,"rb") as bboxes_file:
                bboxes = cPickle.load(bboxes_file)

                
                for boxes in bboxes:
                    
                    # for box in boxes:
                    if boxes != []:
                        # print("null")    
                        new_line = line_function(boxes,fieldnames,counter)

                        # # print(new_line)
                        writer.writerow(new_line)
                    counter += 1
            CURRENT_SCENE += 1

def line_boxes_alt(boxes,fieldnames,counter):
    boxes = [box.tolist() for box in boxes]
    
    line = [boxes]
    line.append(counter)

    new_line = {}
    for i,key in enumerate(fieldnames):
        new_line[key] = line[i]
    return new_line

def cpkl_to_csv(save_path,directories,fieldnames,suffix,line_function):

    global CURRENT_SCENE
    CURRENT_SCENE = 0
    counter = 0
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path,"a") as csv_scene:
        # writer_scene = csv.writer(csv_scene)
        
        writer = csv.DictWriter(csv_scene, fieldnames=fieldnames)
        writer.writeheader()

        for i,dir_ in enumerate(directories):
            bboxes_path = DATA + "/" + dir_ + "/" + dir_ + suffix
                

            with open(bboxes_path,"rb") as bboxes_file:
                bboxes = cPickle.load(bboxes_file)


                for boxes in bboxes:
                    for box in boxes:

                        new_line = line_function(box,fieldnames,counter)

                        # print(new_line)
                        writer.writerow(new_line)
                    counter += 1
            CURRENT_SCENE += 1


def main():

    print("Getting scene lengths...")
    global SCENE_LENGTHS
    SCENE_LENGTHS = get_scene_lengths(DATA)



    print("Done!")
    print("Loading directories names")
    directories = helpers.get_dir_names(DATA,lower = False,ordered = True,descending = False)

    print("Loading boxes from cpkl to csv")

    cpkl_to_csv_alt(ROOT + DATASET +"/" + "boxes.csv",directories,['boxes','frame'],BOXES_SUFFIX,line_boxes_alt)

    print("Loading init from cpkl to csv")

    cpkl_to_csv(ROOT + DATASET +"/" + "init.csv",directories,['trajectory_label', 'class', 'box','frame'],INIT_SUFFIX,line_init)   

    # print("Loading trajectories from cpkl to csv")                
                       
    # cpkl_to_csv(ROOT + DATASET +"/" + "trajectories.csv",directories,['y', 'x', 'label','initial_state','frame'],TRAJECTORIES_SUFFIX,line_trajectory)                   
                     

    print("Done!")



    

 

        
if __name__ == "__main__":
    main()
