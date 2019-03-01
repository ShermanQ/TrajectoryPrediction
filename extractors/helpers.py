import os
import csv


"""
    Get the directories/files contained in a directory
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
    check if file exists and delete it
"""
def remove_file(file):
    if os.path.exists(file):
        os.remove(file)


"""
    In dir dir_path, delete every file
    which name contains one of the strings
    in strings
"""
def del_files_containing_string(strings,dir_path):
    csv_files = get_dir_names(dir_path)
    for csv_ in csv_files:
        for string in strings:
            if string in csv_:
                file_ = dir_path + csv_
                remove_file(file_)


def clip_scene(clips,scene_path,new_path = "./data/temp/temp.csv"):
    x_low,x_up,y_low,y_up = clips
    with open(scene_path) as scene_csv:
        csv_reader = csv.reader(scene_csv)

        with open(new_path,"a") as new_csv:
            csv_writer = csv.writer(new_csv)

            for line in csv_reader:
                x = float(line[4])
                y = float(line[5])

                if x > x_low and x < x_up and y > y_low and y < y_up:
                    csv_writer.writerow(line)
    remove_file(scene_path)

    os.rename(new_path,scene_path)

    remove_file(new_path)

