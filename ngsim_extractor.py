import csv
import os
import time
import extractors.helpers as helpers
import numpy as np
import json
from sklearn import neural_network,preprocessing,metrics,model_selection
import pandas as pd 
import clip_scene


def del_files_containing_string(strings,dir_path):
    # in data/csv delete every csv file related to the selected scene
    csv_files = helpers.get_dir_names(dir_path)
    for csv_ in csv_files:
        for string in strings:
            if string in csv_:
                file_ = dir_path + csv_
                # print(scene_file)
                if os.path.exists(file_):
                    os.remove(file_)

def feet_meters(value,conversion_rate = 0.3048 ):
    return conversion_rate * value

def get_bbox(direction,new_pos,length,width):
    if direction == 2:
        top_left = np.subtract(new_pos, [width/2.,length]).tolist()
        bottom_right = np.subtract(new_pos, [-width/2.,0]).tolist()
    elif direction == 4:
        top_left = np.subtract(new_pos, [width/2.,0]).tolist()
        bottom_right = np.subtract(new_pos, [-width/2.,-length]).tolist()
    elif direction == 3 :
        top_left = np.subtract(new_pos, [0.,-width/2]).tolist()
        bottom_right = np.subtract(new_pos, [-length,width/2]).tolist()
    elif direction == 1 :
        top_left = np.subtract(new_pos, [length,-width/2]).tolist()
        bottom_right = np.subtract(new_pos, [0,width/2]).tolist()
    else: 
        top_left,bottom_right = [-1,-1],[-1,-1]
    return [c for c in top_left+bottom_right]
    
def add_obs(trajectories,row,subscene,dataset,dict_type,trajectory_counter):
    id_,frame,type_ = int(row[0]),int(row[1]),dict_type[row[10]]
    id_ = trajectory_counter
    new_pos = [
                feet_meters(float(row[4])),
                feet_meters(float(row[5]))
            ]
    # new_pos = [
    #             feet_meters(float(row[6])),
    #             feet_meters(float(row[7]))
    #         ]

    # width and length of the observed vehicle
    length = feet_meters(float(row[8]))
    width = feet_meters(float(row[9]))
    # new_pos = [
    #             feet_meters(float(row[6])),
    #             feet_meters(float(row[7]))
    #         ]

    direction = int(row[18])
    bbox = get_bbox(direction,new_pos,length,width)

    
    

    if id_ not in trajectories:

        trajectories[id_] = {
            "coordinates":[new_pos],
            "frames":[frame],
            "type": type_,
            "subscene" : subscene,
            "bboxes" : [bbox],
            "dataset": dataset                                                        
            }
    else:
        

        trajectories[id_]["coordinates"].append(new_pos)
        trajectories[id_]["frames"].append(frame)
        trajectories[id_]["bboxes"].append(bbox)
    return trajectory_counter

def persist_trajectories(trajectories,file_path):
    with open(file_path,"a") as csv_:
        csv_writer = csv.writer(csv_)

        for id_ in trajectories:
            trajectory = trajectories[id_]
            dataset = trajectory["dataset"]
            subscene = trajectory["subscene"]
            type_ = trajectory["type"]

            # model_path = models[subscene]["model_path"]
            # std_path = models[subscene]["std_path"]

            # model = load(model_path)
            # std = load(std_path)
            # coordinates = std.transform(trajectory["coordinates"])
            # coordinates = model.predict(coordinates).tolist()
            # trajectory["coordinates"] = coordinates

            for frame,pos,bbox in zip(trajectory["frames"],trajectory["coordinates"],trajectory["bboxes"]):
                # pos = std.transform([pos])
                # pos = model.predict(pos)[0].tolist()
                row = []
                row.append(dataset) #dataset
                row.append(subscene) #scene
                row.append(frame) # frame
                row.append(id_) # id
                row.append(pos[0]) #x
                row.append(pos[1])  #y
                for b in bbox:
                    row.append(b) 
                row.append(type_)
                csv_writer.writerow(row)
    return {}

def split_ngsim(data_file):
    with open(data_file) as data_reader:
        data_reader = csv.reader(data_reader, delimiter=',')
        for i, line in enumerate(data_reader):               
            if i != 0:
                if line[16] != "0" and line[17] == "0":
                    subscene = line[-1] +"_inter" +line[16]
                    file_path = "./data/datasets/ngsim/" + subscene + ".csv"
                    with open(file_path,"a") as csv_:
                        csv_writer = csv.writer(csv_)
                        csv_writer.writerow(line)


                
                



                

                


def split_ngsim_correspondences(data_file,correspondences):
    with open(data_file) as data_reader:
        data_reader = csv.reader(data_reader, delimiter=',')
        for i, line in enumerate(data_reader):               
            if i != 0:
                scene = line[-1]
                if scene in correspondences:
                    corres_scene = correspondences[scene]
                    inter = line[16]
                    section = line[17]

                    if inter in corres_scene["inters"]:
                        subscenes = corres_scene["inters"][inter]
                    elif section in corres_scene["sections"]:
                        subscenes = corres_scene["sections"][section]
                    else:
                        subscene = []


                    
                    for subscene in subscenes:
                        file_path = "./data/datasets/ngsim/" + subscene + ".csv"
                        with open(file_path,"a") as csv_:
                            csv_writer = csv.writer(csv_)
                            csv_writer.writerow(line)
            





        


        
        
        



                 


    
def main():
    parameters = "parameters/ngsim_extractor.json"
    with open(parameters) as parameters_file:
        parameters = json.load(parameters_file)
    csv_path = "./data/csv/"
    dataset = parameters["dataset"]
    data_file = parameters["data_file"]
    data_dir = parameters["data_dir"]
    # dir_model = parameters["dir_model"]
    # train_files = parameters["train_files"]
    # std_files = parameters["std_files"]
    # feet_meters = parameters["feet_meters"]
    dict_type = parameters["dict_type"]    
    scene_names = parameters["scene_names"] # list of scenes I want  to keep    
    subscene_names = parameters["subscene_names_all"] # list of subscenes I want  to keep
    correspondences = parameters["correspondences"]
    # models_json = parameters["models"]
    clips = parameters["clip_scenes"]



    start = time.time()

    
    del_files_containing_string(scene_names,data_dir) 
    split_ngsim_correspondences(data_file,correspondences)
  
 

    print(time.time() - start)
    # split_ngsim(data_file)
    del_files_containing_string(scene_names,csv_path) 

    trajectories = {}
    trajectory_counter = 0

    

    for subscene in subscene_names:
        data_path = "./data/datasets/ngsim/"+subscene+".csv"
        with open(data_path) as data_reader:
            data_reader = csv.reader(data_reader, delimiter=',')

            last_id = -1
            for line in data_reader:
              
                
                new_id = int(line[0])
                if  last_id != new_id:
                    file_path = csv_path + subscene + ".csv"
                    
                    trajectories = persist_trajectories(trajectories,file_path)
                    trajectory_counter += 1
                    

                trajectory_counter = add_obs(trajectories,line,subscene,dataset,dict_type,trajectory_counter)
                last_id = new_id

        file_path = csv_path + subscene + ".csv"
        trajectories = persist_trajectories(trajectories,file_path)

    for subscene in clips:
        file_path = csv_path + subscene 
        print(file_path)
        clip = clips[subscene]
        print(clip)
        clip_scene.clip_scene(clip[0],clip[1],clip[2],clip[3],file_path)

                            
                            

                            
                                

    print(time.time()-start)
if __name__ == "__main__":
    main()




# def train_unit_converter(train_file,std,model = neural_network.MLPRegressor(hidden_layer_sizes = (10,10)),random_seed = 42):
#     with open(train_file) as data_reader:
#         data_reader = csv.reader(data_reader, delimiter=',')
#         X, Y = [],[]
#         for i, line in enumerate(data_reader):
#             y = [float(line[0]),float(line[1])]
#             x = [float(line[2]),float(line[3])]
#             X.append(x)
#             Y.append(y)
#         X = pd.DataFrame(X)
#         Y = pd.DataFrame(Y)

#         # std = preprocessing.StandardScaler()

        

#         X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=random_seed)

#         X_train = std.transform(X_train)
#         X_test = std.transform(X_test)
#         # model = MLPRegressor(hidden_layer_sizes = (10,10))

#         model = model.fit(X_train,y_train)
#         train_pred,test_pred = model.predict(X_train),model.predict(X_test)
#         train_err = metrics.mean_squared_error(y_train,train_pred)
#         test_err = metrics.mean_squared_error(y_test,test_pred)

#         return model,std,train_err,test_err

# from joblib import dump, load

# def get_std(train_file):
#     points = []
#     # for train_file in train_files:
#     with open(train_file) as data_reader:
#         data_reader = csv.reader(data_reader, delimiter=',')
#         for line in data_reader:
#             x = float(line[0])
#             y = float(line[1])

#             points.append([x,y])
    
#     mms = preprocessing.StandardScaler()
#     mms = mms.fit(points)
#     return mms




# def train_converters(train_files,std_files,train_dir,dir_model,dict_path):
#     model_dict = {}
#     # dict_path = dir_model + "models.json"
#     # std_l = get_std([train_dir+f for f in train_files if "lankershim" in f])
#     # std_p = get_std([train_dir+f for f in train_files if "peachtree" in f])

#     for train_file,std_file in zip(train_files,std_files):
#         std = get_std(train_dir+std_file)
        
#         model,std,train_err,test_err = train_unit_converter(train_dir+train_file,std)
#         model_path = train_file.split("_")
#         model_name = model_path[0]+"_"+model_path[1]
        
#         model_path = dir_model + model_name + "_model.joblib"
#         std_path = dir_model + model_name + "_std.joblib"

        
#         model_dict[model_name] = {
#             "model_path" : model_path,
#             "std_path" : std_path,
#             "train_error": train_err,
#             "test_error": test_err
#         }

#         dump(model, model_path) 
#         dump(std, std_path) 
#     with open(dict_path,"w") as dict_json:
#         json.dump(model_dict,dict_json,indent = 2)

# def get_train_data(data_file ):
#     with open(data_file) as data_reader:
#         data_reader = csv.reader(data_reader, delimiter=',')
#         for i, line in enumerate(data_reader):               
#             if i != 0:
                
#                 if line[16] != "0" and line[17] == "0":
#                     subscene = line[-1] + "_inter" +line[16]
#                     file_path = "./data/datasets/ngsim/" + subscene + "_train.csv"
#                     new_line = [
#                         feet_meters(float(line[4])),
#                         feet_meters(float(line[5])),
#                         feet_meters(float(line[6])),
#                         feet_meters(float(line[7]))
#                     ]
#                     with open(file_path,"a") as csv_:
#                         csv_writer = csv.writer(csv_)
#                         csv_writer.writerow(new_line)

# def get_std_train_data(files,data_dir):
#     for file_ in files:
#         data_file = data_dir + file_ + ".csv"
#         with open(data_file) as data_reader:
#             data_reader = csv.reader(data_reader, delimiter=',')
#             file_path = data_dir + file_ + "_std.csv"
#             with open(file_path,"a") as csv_:
#                 csv_writer = csv.writer(csv_)
                
#                 for i, line in enumerate(data_reader): 
#                     new_line = [
#                             feet_meters(float(line[4])),
#                             feet_meters(float(line[5])),
#                             feet_meters(float(line[6])),
#                             feet_meters(float(line[7]))
#                         ]
#                     csv_writer.writerow(new_line)