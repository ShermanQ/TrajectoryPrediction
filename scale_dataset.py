from prepare_training import digit_manager,scene_scaler
from prepare_training import img_scaler
from prepare_training import samples_manager,prepare_training_hdf5,prepare_training_frames_hdf5
import sys
import json
import csv
import helpers
import h5py
import os

def augment_scene_list(scene_list,angles):
    new_list = []

    for scene in scene_list:
        new_list.append(scene)
        for angle in angles:
            scene_angle = scene + "_{}".format(angle)
            new_list.append(scene_angle)
    return new_list
#python scale_dataset.py parameters/parameters.json parameters/prepare_training.json parameters/data.json parameters/preprocessing.json
def main():
    args = sys.argv

    parameters_path = json.load(open(args[1]))

    prepare_training_params = json.load(open(args[2]))
    
    data_params_path = args[3]
    data = json.load(open(data_params_path))
    preprocessing = json.load(open(args[4]))




    scene_list,train_list,test_list = None, None, None

    prep_toy = prepare_training_params["toy"]

    if prep_toy:
        scene_list = prepare_training_params["toy_scenes"]
        train_list = prepare_training_params["toy_train_scenes"]
        test_list = prepare_training_params["toy_test_scenes"]
    else:
        scene_list = prepare_training_params["selected_scenes"]
        train_list = prepare_training_params["train_scenes"]
        test_list = prepare_training_params["test_scenes"]

        scene_list = augment_scene_list(scene_list,preprocessing["augmentation_angles"])
        train_list = augment_scene_list(train_list,preprocessing["augmentation_angles"])
        test_list = augment_scene_list(test_list,preprocessing["augmentation_angles"])



    print("Managing decimal number")
    nb_digits = int(prepare_training_params["number_digits_meters"])
    digit_man = digit_manager.DigitManager(data_params_path,nb_digits)
    for scene in scene_list:
        print(scene)
        digit_man.change_digit_number(scene)
    print("DOne!")

    if prepare_training_params["normalize"]:
        print("Normalizing scenes")
        center = int(prepare_training_params["center"])
        scaler = None
        if data["multiple_scalers"]:
            print("-----------multiple scalers")
            scaler = scene_scaler.SceneScaler(data_params_path,center)        
        else:
            print("-----------unique scaler")
            scaler = scene_scaler.SceneScalerMultiScene(data_params_path,center,scene_list)
            
        for scene in scene_list:
            print(scene)
            scaler.min_max_scale(scene)
        print("DOne!")


        print("Managing normalized decimal number")
        nb_digits = int(prepare_training_params["number_digits_norm"])
        digit_man = digit_manager.DigitManager(data_params_path,nb_digits)
        for scene in scene_list:
            print(scene)
            digit_man.change_digit_number(scene)
        print("DOne!")


    img_size = int(prepare_training_params["img_size"])
    scaler = img_scaler.ImgScaler(data_params_path,img_size)  
    print("scaling images")
    for scene in scene_list:
        print(scene)
        scaler.scale(scene)

    # sampler = prepare_training_frames_hdf5.PrepareTrainingFramesHdf5(data_params_path,args[2],prep_toy)
    
    # print("sampling frames")
    
    # for scene in scene_list:
    #     print(scene)
    #     sampler.extract_data(scene)
    # print("DOne!")

    # sampler = prepare_training_hdf5.PrepareTrainingHdf5(data_params_path,args[2],prep_toy)
    
    # print("sampling trajectories")
    # for scene in scene_list:
    #     print(scene)

    #     sampler.extract_data(scene)
    #     print("DOne!")


    


    
    
    

if __name__ == "__main__":
    main()