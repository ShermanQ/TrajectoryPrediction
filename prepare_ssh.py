import os
import sys

args = sys.argv

dir_name = args[1]

os.system("mkdir {}".format(dir_name))

os.system("mkdir {}/data".format(dir_name))
os.system("mkdir {}/learning".format(dir_name))

os.system("cp -r {}/parameters {}/parameters".format("master",dir_name))


os.system("mkdir {}/data/reports".format(dir_name))
os.system("mkdir {}/data/reports/gradients".format(dir_name))
os.system("mkdir {}/data/reports/losses".format(dir_name))
os.system("mkdir {}/data/reports/samples".format(dir_name))

os.system("mkdir {}/data/scalers".format(dir_name))
os.system("mkdir {}/data/prepared_datasets".format(dir_name))


os.system("cp -r {}/data/prepared_datasets/images {}/data/prepared_datasets/images".format("master",dir_name))
os.system("cp -r {}/data/scalers/scaler.joblib {}/data/scalers/scaler.joblib".format("master",dir_name))

os.system("cp -r {}/learning/classes {}/learning/classes".format("master",dir_name))
os.system("cp -r {}/learning/helpers {}/learning/helpers".format("master",dir_name))

os.system("mkdir {}/learning/data".format(dir_name))

os.system("mkdir {}/learning/data/models".format(dir_name))



os.system("cp -r {}/learning/data/pretrained_models {}/learning/data/pretrained_models".format("master",dir_name))

os.system("cp -r {}/learning/data/neighbors.json {}/learning/data/neighbors.json".format("master",dir_name))


os.system("cp master/learning/*.py {}/learning/".format(dir_name))

os.system("cp training.sh {}/training.sh".format(dir_name))
os.system("cp master/clean_reports.sh {}/clean_reports.sh".format(dir_name))


print("don't forget to modify training.sh and the parameters of the training")

