import time
import csv
import extractors.helpers as helpers
import os

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

            # scene_csv = CSV_PATH + scene + ".csv"
    
            # if os.path.exists(scene_csv):
            #     os.remove(scene_csv)


            # with open(scene_csv,"a") as csv_scene:
            #     writer_scene = csv.writer(csv_scene)

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
                

                    with open(subscene_path) as a:
                        
                        
                        for z,line in enumerate(a):
                            new_line = helpers.parse_line(line,scene + str(i), DATASET )
                            writer.writerow(new_line)
                            writer_scene.writerow(new_line)
                        # print(z)
                    
    print("Execution time: " + str(time.time() - start) + " s") 

if __name__ == "__main__":
    main()
