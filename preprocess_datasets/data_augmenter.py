import sys
import json
import csv
import numpy as np
import pandas as pd
import helpers
from scipy.spatial.distance import euclidean
from skimage import io,transform,util
import os



class DataAugmenter():
    def __init__(self,preprocessing_params,data,prepare_params):
        preprocessing_params = json.load(open(preprocessing_params)) 
        prepare_params = json.load(open(prepare_params)) 
        data = json.load(open(data))


        # self.scenes = preprocessing_params["scenes"]
        self.destination_dir = data["preprocessed_datasets"]
        self.original_file = data["filtered_datasets"] + "{}.csv"
        self.angles = preprocessing_params["augmentation_angles"]
        self.augmented_name = "{}{}_{}.csv"

        self.original_image = data["original_images"] + "{}.jpg"
        self.destination_image = data["preprocessed_images"] + "{}{}.jpg"
        
    def augment_scene(self,scene):
        helpers.remove_file("{}{}.csv".format(self.destination_dir,scene))
        command = "cp {} {}".format(self.original_file.format(scene),"{}{}.csv".format(self.destination_dir,scene))
        
        os.system(command)  

        for angle in self.angles:
            print("{} degrees".format(angle))

            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            
            r = np.array([[c,-s],
                          [s,c]
                         ])

            
            with open(self.original_file.format(scene)) as original_file:
                original_reader = csv.reader(original_file)

                helpers.remove_file(self.augmented_name.format(self.destination_dir,scene,angle))
            
                with open(self.augmented_name.format(self.destination_dir,scene,angle), "a+") as new_file:
                    writer= csv.writer(new_file)

                    for orig_row in original_reader:
                        points = [float(e) for e in orig_row[4:10]]
                        points = np.array([points[0:2],points[2:4],points[4:]])  
                        points = points.reshape(len(points),len(points[0]))
                        r_points = np.dot(r, np.transpose(points))
                        
                        r_points = np.transpose(r_points).flatten()
                        new_row = orig_row
                        new_row[4:10] = r_points
                        writer.writerow(new_row)

    def augment_scene_images(self,scene):
        img = io.imread(self.original_image.format(scene))

        io.imsave(self.destination_image.format(scene,""),img)


        for angle in self.angles:
            r_img = transform.rotate(img,angle)
            io.imsave(self.destination_image.format(scene,"_{}".format(angle)),r_img)








        


        
# python preprocess_datasets/stops_remover.py parameters/preprocessing.json parameters/data.json
def main():
    args = sys.argv
    stops = StopsRemover(args[1],args[2],args[3])
    # stats.get_stats()



if __name__ == "__main__":
    main()