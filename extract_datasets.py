from extractors import highd_extractor,koper_extractor,ngsim_extractor,sdd_extractor

import sys
import json
import csv
import helpers
import h5py
import os

#python extract_datasets.py parameters/parameters.json parameters/prepare_training.json parameters/data.json
def main():
    args = sys.argv

    parameters_path = json.load(open(args[1]))

    prepare_training_params = json.load(open(args[2]))
    
    data_params_path = args[3]
    data = json.load(open(data_params_path))


    extractor_list = [
        highd_extractor.HighdExtractor(data_params,parameters_path["highd_extractor"]),
        koper_extractor.KoperExtractor(data_params,parameters_path["koper_extractor"]),
        ngsim_extractor.NgsimExtractor(data_params,parameters_path["ngsim_extractor"]),
        sdd_extractor.SddExtractor(data_params,parameters_path["sdd_extractor"])
    ]

    for extractor in extractor_list:
        extractor.extract()

    

    
    
    

if __name__ == "__main__":
    main()