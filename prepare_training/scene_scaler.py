import os 
import csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import helpers
import json
import sys
import joblib


class SceneScaler():
    def __init__(self,data,center):
        data = json.load(open(data))

        self.center = center
        self.temp = data["temp"] + "temp.csv"
        self.original_file = data["preprocessed_datasets"] + "{}.csv"
        self.scaler_dest = data["scalers"] + "{}.joblib"


    def min_max_scale(self,scene):

        # self.original_file = self.original_file.format(scene)

        helpers.remove_file(self.temp)
        os.rename(self.original_file.format(scene),self.temp)
        helpers.remove_file(self.original_file.format(scene))

        with open(self.original_file.format(scene),"a+") as data_csv:
            data_writer = csv.writer(data_csv)

            mms = MinMaxScaler()


            min_x,max_x,min_y,max_y = self.__get_boudaries(self.temp)
            
            print(min_x,max_x,min_y,max_y)
            x_mean = (min_x + max_x)/2.0
            y_mean = (min_y + max_y)/2.0

            min_ = min(min_x - x_mean,min_y - y_mean) 
            max_ = max(max_x - x_mean,max_y  - y_mean)

            # print(min_,max_)
            mms = mms.fit([[min_],[max_]])
            

            with open(self.temp) as scene_csv:
                data_reader = csv.reader(scene_csv)
                for row in data_reader:
                    if self.center:
                        row = self.__center_scene(row,x_mean,y_mean)
                    new_row = row
                    
                    ps_untransformed = [[float(row[i])] for i in range(4,10)]
                    ps = mms.transform(ps_untransformed)

                    
                    for i in range(len(ps)):
                        if ps_untransformed[i][0] == -10000:
                            new_row[4 + i] = -1
                        else:
                            new_row[4 + i] = ps[i][0]
                    data_writer.writerow(new_row)
        helpers.remove_file(self.temp)

        
        helpers.remove_file(self.scaler_dest.format(scene))
        joblib.dump(mms, self.scaler_dest.format(scene)) 

    def __get_boudaries(self,file_path):
        with open(file_path) as scene_csv:
            data_reader = csv.reader(scene_csv)

            min_x,min_y = 10e30,10e30
            max_x,max_y = 10e-30,10e-30
            
            for row in data_reader:
                
                # x = np.min([[float(row[4])],[float(row[6])],[float(row[8])]])
                # y = np.min([[float(row[5])],[float(row[7])],[float(row[9])]])
                # print([[float(row[i])] for i in range(4,10,2) ])
                # print([[float(row[i])] for i in range(5,11,2) ])

                x = np.min([[float(row[i])] for i in range(4,10,2) if float(row[i]) != -10000])
                y = np.min([[float(row[i])] for i in range(5,11,2) if float(row[i]) != -10000])

                if x < min_x and x != -1:
                    min_x = x
                if y < min_y and y != -1:
                    min_y = y
                if x > max_x and x != -1:
                    max_x = x
                if y > max_y and y != -1:
                    max_y = y
        return min_x,max_x,min_y,max_y

    def __center_scene(self,row,x_mean,y_mean):
        new_row = row
        for i in range(4,10,2):
            if float(row[i]) != -10000:
                new_row[i] = float(row[i]) - x_mean
        for i in range(5,11,2):
            if float(row[i]) != -10000:
                new_row[i] = float(row[i]) - y_mean
        return new_row           


class SceneScalerMultiScene():
    def __init__(self,data,center,scene_list):
        data = json.load(open(data))

        self.center = center
        self.temp = data["temp"] + "temp.csv"
        self.original_file = data["preprocessed_datasets"] + "{}.csv"
        self.scaler_dest = data["scalers"] 
        self.scene_list = scene_list
        self.scaler = None

    def __get_scaler(self):
        mms = MinMaxScaler()

        min_ = 1e30
        max_ = -1e30
        for scene in self.scene_list:

            min_x,max_x,min_y,max_y = self.__get_boudaries(self.original_file.format(scene))
            
            # print(min_x,max_x,min_y,max_y)
            x_mean = (min_x + max_x)/2.0
            y_mean = (min_y + max_y)/2.0

            min_scene = min(min_x - x_mean,min_y - y_mean) 
            max_scene = max(max_x - x_mean,max_y  - y_mean)

            print(min_scene,max_scene)

            min_ = min(min_scene,min_)
            max_ = max(max_scene,max_)
        print(min_,max_)

        mms = mms.fit([[min_],[max_]])
        print(mms.data_min_,mms.data_max_)
        self.scaler = mms

        helpers.remove_file(self.scaler_dest)
        joblib.dump(self.scaler, self.scaler_dest) 




    def min_max_scale(self,scene):
        if self.scaler == None:
            self.__get_scaler()

        helpers.remove_file(self.temp)
        os.rename(self.original_file.format(scene),self.temp)
        helpers.remove_file(self.original_file.format(scene))

        

        with open(self.original_file.format(scene),"a+") as data_csv:
            data_writer = csv.writer(data_csv)

            min_x,max_x,min_y,max_y = self.__get_boudaries(self.temp)
            
            x_mean = (min_x + max_x)/2.0
            y_mean = (min_y + max_y)/2.0

            with open(self.temp) as scene_csv:
                data_reader = csv.reader(scene_csv)
                for row in data_reader:
                    if self.center:
                        row = self.__center_scene(row,x_mean,y_mean)
                    new_row = row
                    
                    ps_untransformed = [[float(row[i])] for i in range(4,10)]
                    ps = self.scaler.transform(ps_untransformed)

                    
                    for i in range(len(ps)):
                        if ps_untransformed[i][0] == -10000:
                            new_row[4 + i] = -1
                        else:
                            new_row[4 + i] = ps[i][0]
                    data_writer.writerow(new_row)
        helpers.remove_file(self.temp)

        
    def __get_boudaries(self,file_path):
        with open(file_path) as scene_csv:
            data_reader = csv.reader(scene_csv)

            min_x,min_y = 10e30,10e30
            max_x,max_y = 10e-30,10e-30
            
            for row in data_reader:
                
                # x = np.min([[float(row[4])],[float(row[6])],[float(row[8])]])
                # y = np.min([[float(row[5])],[float(row[7])],[float(row[9])]])
                # print([[float(row[i])] for i in range(4,10,2) ])
                # print([[float(row[i])] for i in range(5,11,2) ])

                x = np.min([[float(row[i])] for i in range(4,10,2) if float(row[i]) != -10000])
                y = np.min([[float(row[i])] for i in range(5,11,2) if float(row[i]) != -10000])

                if x < min_x and x != -1:
                    min_x = x
                if y < min_y and y != -1:
                    min_y = y
                if x > max_x and x != -1:
                    max_x = x
                if y > max_y and y != -1:
                    max_y = y
        return min_x,max_x,min_y,max_y

    def __center_scene(self,row,x_mean,y_mean):
        new_row = row
        for i in range(4,10,2):
            if float(row[i]) != -10000:
                new_row[i] = float(row[i]) - x_mean
        for i in range(5,11,2):
            if float(row[i]) != -10000:
                new_row[i] = float(row[i]) - y_mean
        return new_row     

# python prepare_training/scene_scaler.py parameters/data.json 1 lankershim_inter2
def main():

    args = sys.argv

    scene_scaler = SceneScaler(args[1],int(args[2]))
    scene_scaler.min_max_scale(args[3])

    



        


        





                    
if __name__ == "__main__":
    main()