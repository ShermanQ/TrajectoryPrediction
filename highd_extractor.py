import csv
import os
import time
import extractors.helpers as helpers
import numpy as np



DATASETS = "./data/datasets/"
DATASET = "highd"
DATAFILE = DATASETS + DATASET + "/" + DATASET + ".csv"
CSV = "./data/csv/"
MAIN = "main/"
TRACKS = "/tracks/"
META = "/tracks_meta/"

def main():
    csv_files = helpers.get_dir_names(DATASETS + DATASET+TRACKS)
    csv_meta = helpers.get_dir_names(DATASETS + DATASET+META,lower = False)
    print(csv_files)

    # for csv_ in csv_files:
    #     scene_file = CSV + csv_
    #     # print(scene_file)
    #     if os.path.exists(scene_file):
    #         os.remove(scene_file)


    dataset_file = CSV + MAIN + DATASET + ".csv"
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    with open(dataset_file,"a") as dataset_csv:
        
        dataset_writer = csv.writer(dataset_csv)

        for csv_,meta in zip(csv_files,csv_meta):
            print("processing scene: " + csv_)
            with open(DATASETS+DATASET+TRACKS + csv_) as csv_reader:
                csv_reader = csv.reader(csv_reader, delimiter=',')

                with open(DATASETS+DATASET+META + meta) as meta_reader:
                    meta_reader = csv.reader(meta_reader, delimiter=',')
                
                    scene_file = CSV + csv_
                    if os.path.exists(scene_file):
                        os.remove(scene_file)
                    with open(scene_file,"a") as scene_csv:
                        subscene_writer = csv.writer(scene_csv)

                        line1 = next(meta_reader)
                        last_id = "-2"
                        for i,line in enumerate(csv_reader):
                            if i != 0:
                                new_id = line[1]
                                if new_id != last_id:
                                    try:
                                        line1 = next(meta_reader)
                                    except:
                                        "no line available"
                                new_scene_name = csv_.split(".")[0]
                                unknown = -1
                                row = []
                                row.append(DATASET) #dataset
                                row.append(new_scene_name) #scene
                                row.append(int(line[0])) # frame
                                row.append(int(line[1])) # id
                                row.append(float(line[2])) #x
                                row.append(float(line[3]))  #y

                                row.append(unknown) #xl
                                row.append(unknown) #yl
                                row.append(unknown) #xb
                                row.append(unknown) #yb

                                row.append(line1[6].lower()) # type

                                subscene_writer.writerow(row)
                                dataset_writer.writerow(row)
                    

if __name__ == "__main__":
    main()