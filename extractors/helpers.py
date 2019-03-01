import os



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