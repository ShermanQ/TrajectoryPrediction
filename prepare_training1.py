import csv
from itertools import islice
import extractors.helpers as helpers 
import json
import numpy as np
import time
from itertools import tee
import os

def persist(observations,data_writer,label_writer):
    pass

def main():
    parameters = "parameters/prepare_training.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    s = time.time()
    helpers.extract_frames(parameters["original_file"],parameters["frames_temp"],save = True)

    print(time.time()-s)

    
    if os.path.exists(parameters["data_path"]):
        os.remove(parameters["data_path"])
    if os.path.exists(parameters["label_path"]):
        os.remove(parameters["label_path"])
    print(time.time()-s)

    with open(parameters["data_path"],"a") as data_csv:
        data_writer = csv.writer(data_csv)
        with open(parameters["label_path"],"a") as label_csv:
            label_writer = csv.writer(label_csv)

            with open(parameters["frames_temp"]) as frames:
                observations = {}
                sample_id = 0
                for frame in frames:
                    delete_ids = []
                    observations[sample_id] = []
                    sample_id += 1

                    for id_ in observations:
                        if len(observations[id_]) < parameters["t_obs"] + parameters["t_pred"]:
                            observations[id_].append(frame)
                        else:
                            persist(observations[id_],data_writer,label_writer)
                            delete_ids.append(id_)
                    for id_ in delete_ids:
                        del observations[id_]

                    
                    print(time.time()-s)
    print(time.time()-s)




    os.remove(parameters["frames_temp"])

                    
if __name__ == "__main__":
    main()