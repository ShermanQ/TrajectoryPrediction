from scipy.spatial.distance import euclidean
import json
import helpers
import os
import csv

class Pixel2Meters():
    def __init__(self,data,pixel2meters):
        data = json.load(open(data))
        self.correspondences = json.load(open(data["sdd_pixel2meters"]))
        self.original_image = data["original_images"] + "{}.jpg"
        self.destination_image = data["prepared_images"] + "{}.jpg"
        self.temp = data["temp"] + "temp.csv"
        self.original_file = data["filtered_datasets"] + "{}.csv"
        # self.destination_file = data["filtered_datasets"] + "{}.csv"

        self.pixel2meters = pixel2meters


    def convert(self,scene):
        self.__get_factor(scene)
        helpers.remove_file(self.temp)
        os.rename(self.original_file.format(scene),self.temp)
        helpers.remove_file(self.original_file.format(scene))

        with open(self.original_file.format(scene),"a+") as data_csv:
            data_writer = csv.writer(data_csv)

            with open(self.temp) as scene_csv:
                data_reader = csv.reader(scene_csv)
                for row in data_reader:
                    new_row = row
                    new_coords = None
                    if self.pixel2meters:
                        new_coords = [self.pixel2meter_ratio * float(row[i]) for i in range(4,10)]

                    else :
                        new_coords = [self.meter2pixel_ratio * float(row[i]) for i in range(4,10)]

                    for i,c in enumerate(new_coords):
                        new_row[i+4] = c 
                    data_writer.writerow(new_row)
        
        helpers.remove_file(self.temp)
                    



    def __get_factor(self,scene):
        row = self.correspondences[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        self.pixel2meter_ratio = meter_dist/float(pixel_dist)
        self.meter2pixel_ratio = float(pixel_dist)/meter_dist