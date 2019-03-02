import csv
import json

def min_time(filepath):
    min_ = 10e30
    with open(filepath) as file_csv:
        csv_reader = csv.reader(file_csv)
        for line in csv_reader:
            time_ = int(line[3])
            if time_ < min_:
                min_ = time_
    return min_

def main():
    parameters = "parameters/ngsim_extractor.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    data_dir = parameters["data_dir"]
    subscene_names = parameters["subscene_names_all"]

    dict_min = {}
    for subscene in subscene_names:
        filepath = data_dir + subscene + ".csv"
        min_ = min_time(filepath)
        dict_min[subscene] = min_
    print(dict_min)

if __name__ == "__main__":
    main()
