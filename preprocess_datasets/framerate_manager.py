import sys 
import os
import helpers
import json
import csv

class FramerateManager():
    def __init__(self,data,new_rate):
        data = json.load(open(data))

        self.framerates_json = json.load(open(data["framerates_json"]))
        self.new_rate = new_rate
        self.temp = data["temp"] + "temp.txt"
        self.destination_file = data["filtered_datasets"] + "{}.csv"
        self.original_file = data["extracted_datasets"] + "{}.csv"

    def change_rate(self,scene_name):
        
        
        former_rate = float(self.framerates_json[scene_name])
        
        rate_ratio = int(former_rate/self.new_rate)

        # self.destination_file = self.destination_file.format(scene_name)
        
        helpers.remove_file(self.destination_file.format(scene_name))

        self.counter = 0

        helpers.extract_frames(self.original_file.format(scene_name),self.temp,save=True)
        with open(self.temp) as frames:
            for frame in frames:
                frame = json.loads(frame)
                i = frame["frame"]
                if i % rate_ratio == 0:
                    self.__write_frame(frame["ids"],scene_name)
                    self.counter += 1   
        helpers.remove_file(self.temp) 
        

        
    def __write_frame(self,frame,scene):

        with open(self.destination_file.format(scene),"a") as file_:
            file_writer = csv.writer(file_)
            for key in frame:
                new_line = []
                new_line.append(frame[key]["dataset"])
                new_line.append(frame[key]["scene"])
                new_line.append(self.counter)
                new_line.append(key)
                for e in frame[key]["coordinates"]:
                    new_line.append(e)

                for e in frame[key]["bbox"]:
                    new_line.append(e)

                new_line.append(frame[key]["type"].split("\n")[0])


                file_writer.writerow(new_line)

#python prepare_training/framerate_manager.py parameters/data.json 2.5 lankershim_inter2
def main():

    args = sys.argv
    framerate_manager = FramerateManager(args[1],float(args[2]))

    framerate_manager.change_rate(args[3])
  
if __name__ == "__main__":
    main()