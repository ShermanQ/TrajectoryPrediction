import os
import helpers
# import cPickle
import csv
import numpy as np

DATASET = "bad"
ROOT = "./datasets/"
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
def detections_to_csv(save_path,trajectories_files,scene_lengths):
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path,"a") as csv_file:
        writer = csv.writer(csv_file)
        for i,trajectory_file in enumerate(trajectories_files):
            with open(trajectory_file,"rb") as detections_file:
                frames = cPickle.load(detections_file)
                for frame in frames:
                    for detection in frame:
                        line = []
                        for key in detection:
                            line.append(detection[key])
                        local_frame = int(line[-1])
                        global_frame = local_frame + np.sum(scene_lengths[:i])
                        line[-1] = int(global_frame)
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
                    row[-1] = int(float(row[-1])/ratio)
                    writer.writerow(row) 


def main():

    ### Extract cpkl trajectories to csv ###
    # save_path = ROOT + DATASET +"/" + "detections.csv"

    # directories = helpers.get_dir_names(DATA,lower = False,ordered = True,descending = False)
    # boxes_files = [DATA + dir_ + "/" + dir_ + BOXES_SUFFIX for dir_ in directories]
    # scene_lengths = get_scene_lengths(boxes_files)

    # trajectories_files = [DATA + dir_ + "/" + dir_ + TRAJECTORIES_SUFFIX for dir_ in directories]
    
    # detections_to_csv(save_path,trajectories_files,scene_lengths)
    ####################################

    ### Reduce framerate ###
    framerate = 30.
    new_rate = 10.
    detection_path = ROOT + DATASET +"/" + "detections.csv"
    # detection_sampled_path = ROOT + DATASET +"/" + "detections_sampled"
    # detection_sampled_path += "_" + str(int(framerate))+"to"+ str(int(new_rate))+".csv"
    detection_sampled_path = ROOT + DATASET +"/"+"trajectories.csv"
    reduce_observations_framerate(framerate,new_rate,detection_path,detection_sampled_path)
    ####################################
    
    
if __name__ == "__main__":
    main()
