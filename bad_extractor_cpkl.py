import os
import extractors.helpers as helpers
import cPickle
import csv
import numpy as np
from skimage.transform import ProjectiveTransform


DATASET = "bad"
ROOT = "./data/datasets/"
DATA = ROOT + DATASET + "/data/"
BOXES_SUFFIX = "_bboxes.cpkl"
TRAJECTORIES_SUFFIX = "_trajectories.cpkl"
INIT_SUFFIX = "_init_labels.cpkl"

"""
    In: boxes_files_names: list of relative paths to the bboxes files
    Out: a list containing the number of frame per scene
    Go through every directory in bad dataset and compute the number of frames in it
    Can only be run with python2

"""
def get_scene_lengths(boxes_files_names):
    scene_lengths = []
    for file_name in boxes_files_names:
        with open(file_name,"rb") as bboxes_file:
            bboxes = cPickle.load(bboxes_file)
            scene_lengths.append(len(bboxes))
    return scene_lengths

"""
    In: 
        save_path: csv file to save the extracted detections
        trajectories_files: list of relative paths to the trajectories files
        scene_lengths: a list containing the number of frame per scene

    Summarize every detections in the save_path csv_file
    Can only be run with python2

"""
def detections_to_csv(save_path,trajectories_files,scene_lengths,height,width,homography,center = [878,444]):
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path,"a") as csv_file:
        writer = csv.writer(csv_file)
        transformer = ProjectiveTransform(matrix = homography)
        for i,trajectory_file in enumerate(trajectories_files):
            with open(trajectory_file,"rb") as detections_file:
                frames = cPickle.load(detections_file)
                for frame in frames:
                    for detection in frame:
                        line = []
                        p = [float(detection["x"]) * width,float(detection["y"]) * height]
                        p = np.subtract(p,center).tolist()
                        p[1] *= -1.

                        p = transformer.inverse(p)[0]
                        x = p[0]
                        y = p[1]
                        

                        label = detection["cls_label"]
                        state = detection["is_initial_state"]
                        local_frame = int(detection["t"])
                        

                        global_frame = local_frame + np.sum(scene_lengths[:i])
                        line.append(x)
                        line.append(y)
                        line.append(label)
                        line.append(state)
                        line.append(int(global_frame))
                        line.append("minute number: " + str(i))
                        writer.writerow(line)

"""
    IN:
        framerate: old framerate
        new_rate: new frame rate
        detection_path: path to csv file where trajectories observation were put
        detection_sampled_path: target file to store resampled observations

        save all observations where init = True and one over new_rate/framerate frame
"""
def reduce_observations_framerate(framerate,new_rate,detection_path,detection_sampled_path):
    ratio = int(framerate / new_rate)   

    if os.path.exists(detection_sampled_path):
        os.remove(detection_sampled_path)

    with open(detection_sampled_path,"a") as csv_file:
        writer = csv.writer(csv_file)
        with open(detection_path) as detection_file:
            reader = csv.reader(detection_file)
            for row in reader:
                if row[3] == 'True' or int(row[4]) % ratio == 0:
                    # row[-1] = int(float(row[-1])/ratio)
                    row[-2] = int(float(row[-2])/ratio)

                    writer.writerow(row) 


def main():
    height,width = 720, 1280

    # theta = 1.156657776318685
    # R = [[np.cos(theta),-np.sin(theta)],
    #     [np.sin(theta),np.cos(theta)]
    # ]

    # homography = np.loadtxt("/home/laurent/Documents/master/extractors/datasets/bad/homography/homography.txt")
    # homography = np.loadtxt("/home/laurent/Documents/master/extractors/datasets/bad/homography/homography.txt")
    homography = np.loadtxt("./data/datasets/bad/homography/homography.txt")

    

    ### Extract cpkl trajectories to csv ###
    save_path = ROOT + DATASET +"/" + "detections.csv"

    directories = helpers.get_dir_names(DATA,lower = False,ordered = True,descending = False)
    # directories = [directories[0]]
    # directories = directories[:15]
   
    boxes_files = [DATA + dir_ + "/" + dir_ + BOXES_SUFFIX for dir_ in directories]
    scene_lengths = get_scene_lengths(boxes_files)

    trajectories_files = [DATA + dir_ + "/" + dir_ + TRAJECTORIES_SUFFIX for dir_ in directories]
    
    detections_to_csv(save_path,trajectories_files,scene_lengths,height,width,homography)
    ####################################

    # # ### Reduce framerate ###
    framerate = 30.
    new_rate = 30.
    detection_path = ROOT + DATASET +"/" + "detections.csv"
    # detection_sampled_path = ROOT + DATASET +"/" + "detections_sampled"
    # detection_sampled_path += "_" + str(int(framerate))+"to"+ str(int(new_rate))+".csv"
    detection_sampled_path = ROOT + DATASET +"/"+"trajectories.csv"
    reduce_observations_framerate(framerate,new_rate,detection_path,detection_sampled_path)
    ####################################
    return
    
if __name__ == "__main__":
    main()
