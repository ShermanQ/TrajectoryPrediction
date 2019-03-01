import time
import csv
import extractors.helpers as helpers
import os



def parse_line(line,scene, dataset ):

    dict_type = {
        "Biker" : "bicycle", 
        "Pedestrian" : "pedestrian", 
        "Cart" : "cart", 
        "Car": "car", 
        "Bus" : "bus", 
        "Skater": "skate" }
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

    new_line.append(dict_type[line[9]]) # label type of agent    

    return new_line
#
DATA_PATH = "./data/datasets/sdd/trajectories/"
DATA_FILES = "/annotations.txt"
DATASET = "sdd"
CSV_PATH = "./data/csv/"
MAIN = "main/"

"""
    Parse SDD line by line and return new csv files
    One csv file by scene (including its subscenes)
    One global csv file (including every scene)
"""

def main():

    print("Starting extraction")
    start = time.time()

    main_scene_names = helpers.get_dir_names(DATA_PATH, lower = False)


    sdd_csv = CSV_PATH + MAIN + DATASET + ".csv"
    
    if os.path.exists(sdd_csv):
        os.remove(sdd_csv)

    with open(sdd_csv,"a") as csv_file:
        writer = csv.writer(csv_file)
        
        for scene in main_scene_names:

            print("Processing scene: " + scene)

            subscene_names = helpers.get_dir_names(DATA_PATH + scene)
            
            for i,sub in enumerate(subscene_names):
                print("------subscene: " + scene + str(i))
                
                subscene_path = DATA_PATH + scene + "/" + sub + DATA_FILES
                # print(subscene_path)

                scene_csv = CSV_PATH + scene + str(i) + ".csv"

                if os.path.exists(scene_csv):
                    os.remove(scene_csv)

                with open(scene_csv,"a") as csv_scene:
                    writer_scene = csv.writer(csv_scene)
                

                    with open(subscene_path) as subscene_csv:
                        csv_reader = csv.reader(subscene_csv)
                        
                        
                        for line in csv_reader:
                            new_line = parse_line(line,scene + str(i), DATASET )
                            writer.writerow(new_line)
                            writer_scene.writerow(new_line)
                        # print(z)
                    
    print("Execution time: " + str(time.time() - start) + " s") 

if __name__ == "__main__":
    main()
