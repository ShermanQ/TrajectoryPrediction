import os 
import csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_boudaries(file_path):
    with open(file_path) as scene_csv:
        data_reader = csv.reader(scene_csv)

        min_x,min_y = 10e30,10e30
        max_x,max_y = 10e-30,10e-30
        
        for row in data_reader:
            x = np.min([[float(row[4])],[float(row[6])],[float(row[8])]])
            y = np.min([[float(row[5])],[float(row[7])],[float(row[9])]])

            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
    return min_x,max_x,min_y,max_y
#
def min_max_scale(parameters):
    if os.path.exists(parameters["scene_dest"]):
        os.remove(parameters["scene_dest"])

    with open(parameters["scene_dest"],"a+") as data_csv:
        data_writer = csv.writer(data_csv)

        mms_x = MinMaxScaler()
        mms_y = MinMaxScaler()


        min_x,max_x,min_y,max_y = get_boudaries(parameters["scene"])

        mms_x = mms_x.fit([[min_x],[max_x]])
        mms_y = mms_y.fit([[min_y],[max_y]])

        with open(parameters["scene"]) as scene_csv:
            data_reader = csv.reader(scene_csv)
            for row in data_reader:
                xs = [[float(row[4])],[float(row[6])],[float(row[8])]]
                xs = mms_x.transform(xs)
                ys = [[float(row[5])],[float(row[7])],[float(row[9])]]
                ys = mms_y.transform(ys)
                row[4] = xs[0][0]
                row[5] = ys[0][0]
                row[6] = xs[1][0]
                row[7] = ys[1][0]
                row[8] = xs[2][0]
                row[9] = ys[2][0]
                data_writer.writerow(row)
def main():
    parameters = {
        "scene" : "./data/csv/new_rates/lankershim_inter2_10.0to1.0.csv",
        "scene_dest" : "./data/csv/normalized/lankershim_inter2_10.0to1.0.csv"
    }
    min_max_scale(parameters)
    



        


        





                    
if __name__ == "__main__":
    main()