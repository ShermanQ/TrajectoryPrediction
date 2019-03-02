import os
import csv
import json


""" 
    check if file exists and delete it
"""
def remove_file(file):
    if os.path.exists(file):
        os.remove(file)

"""
    Input: Standardized file_path
    Output a dictionnary of trajectories:
    {
        traj_id: {
            coordinates: [],
            bboxes: [],
            frames: [],
            scene:

        }
    }
"""

def extract_trajectories(file_name,destination_path = "", save = False):

    trajectories = {}

    with open(file_name) as file_:
        file_ = csv.reader(file_, delimiter=',')
        for line in file_:

            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])

            if id_ not in trajectories:

                trajectories[id_] = {
                    "coordinates" : [],
                    "bboxes" : [],
                    "frames" : [],
                    "scene" : line[1],
                    "user_type" : line[10],
                    "id" : id_,
                    "dataset" : line[0]
                }
            trajectories[id_]["coordinates"].append(coordinates)
            trajectories[id_]["bboxes"].append(bbox)
            trajectories[id_]["frames"].append(frame)

    if save:

        remove_file(destination_path)
        
        dict_frame = reindex_frames(file_name)

        with open(destination_path,"a") as scene_txt:
            for key in trajectories:
                
                new_frames = []
                for frame in trajectories[key]["frames"]:
                    new_frames.append(dict_frame[frame])
                trajectories[key]["frames"] = new_frames
                line = trajectories[key]
                # trajectories["id"] = key
                line = json.dumps(line)
                # print(line)
                # print("------")
                scene_txt.write(line + "\n" )
    else:
        return trajectories
    return

"""
    Input: Standardized file_path
    Output a dictionnary of frames:
    {
        frame: {
            object_id : {
                coordinates : [],
                bbox : []
            }

        }
    }
"""


def extract_frames(file_path,destination_path = "", save = False):
    frames = {}

    
    with open(file_path) as file_:
        csv_reader = csv.reader(file_)
        for line in csv_reader:
            # line = line.split(",")
            
            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])
            
            type_ = line[10]
            

            if frame not in frames:
                frames[frame] = {"ids":{}}
    
            frames[frame]["ids"][id_] = {
                "coordinates" : coordinates,
                "bbox" : bbox,
                "type" : type_,
                "scene" : line[1],
                "dataset" : line[0]

                }
            # if save:
            #     frames[frame]["frame"] = frame


        if save:

            remove_file(destination_path)
            

            # current_frame = 0
            # dict_frame = {}

            dict_frame = reindex_frames(file_path)

            with open(destination_path,"a") as scene_txt:
                for key in sorted(frames):

                    # if key not in dict_frame:
                    #     dict_frame[key] = current_frame
                    #     current_frame += 1

                    line = frames[key]
                    line["frame"] = dict_frame[key]
                    # line["frame"] = key
                    line = json.dumps(line)
                    # print(line)
                    # print("------")
                    scene_txt.write(line + "\n" )
        else:
            return frames
    return





def reindex_frames(file_path):
    frames = {}
    with open(file_path) as file_:
        csv_reader = csv.reader(file_)
        for line in csv_reader:
           
            frame = int(line[2])
           

            if frame not in frames:
                frames[frame] = -1
    
        current_frame = 0
        dict_frame = {}

        for key in sorted(frames):
            if key not in dict_frame:
                dict_frame[key] = current_frame
                current_frame += 1
    return dict_frame