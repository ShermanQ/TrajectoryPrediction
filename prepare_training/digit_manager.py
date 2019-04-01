import sys 
import os
import helpers
import json
import csv


class DigitManager():
    def __init__(self,data,digit_number):
        data = json.load(open(data))

        self.framerates_json = json.load(open(data["framerates_json"]))
        self.digit_number = digit_number
        self.temp = data["temp"] + "temp.csv"
        self.destination_file = data["preprocessed_datasets"] + "{}.csv"
        self.original_file = data["preprocessed_datasets"] + "{}.csv"

    def change_digit_number(self,scene_name):

        
        # self.original_file = self.original_file.format(scene_name)
        # self.destination_file = self.destination_file.format(scene_name)

        helpers.remove_file(self.temp)
        os.rename(self.original_file.format(scene_name),self.temp)

        helpers.remove_file(self.original_file.format(scene_name))

        with open(self.temp) as scene_csv:
            csv_reader = csv.reader(scene_csv)
            with open(self.original_file.format(scene_name),"a") as new_csv:
                csv_writer = csv.writer(new_csv)
            
                for row in csv_reader:
                    new_row = row
                    for i in range(4,10):
                        new_row[i] = self.__round_coordinates(float(row[i]))
                    csv_writer.writerow(new_row)
        helpers.remove_file(self.temp)

        
    def __round_coordinates(self,point):
        point = int( point * 10**self.digit_number)/float(10**self.digit_number)
        return point  

#python prepare_training/digit_manager.py parameters/data.json 2 lankershim_inter2       
   
def main():

    args = sys.argv
    digit_manager = DigitManager(args[1],float(args[2]))

    digit_manager.change_digit_number(args[3])
  
if __name__ == "__main__":
    main()