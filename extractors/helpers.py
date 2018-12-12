import os

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