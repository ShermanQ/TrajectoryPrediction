from prepare_training import prepare_training_hdf5,prepare_training_frames_hdf5
import sys
import json
import csv
import helpers
import h5py
import os


#python prepare_training.py parameters/parameters.json parameters/prepare_training.json parameters/data.json parameters/preprocessing.json
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

        scene_list = helpers.augment_scene_list(scene_list,preprocessing["augmentation_angles"])
        train_list = helpers.augment_scene_list(train_list,preprocessing["augmentation_angles"])
        test_list = helpers.augment_scene_list(test_list,preprocessing["augmentation_angles"])



 

    sampler = prepare_training_frames_hdf5.PrepareTrainingFramesHdf5(data_params_path,args[2],prep_toy,smooth = False)
    
    print("sampling frames")
    
    for scene in scene_list:
        print(scene)
        sampler.extract_data(scene)
    print("DOne!")

    # sampler = prepare_training_hdf5.PrepareTrainingHdf5(data_params_path,args[2],prep_toy,smooth = False)
    
    # print("sampling trajectories")
    # for scene in scene_list:
    #     print(scene)

    #     sampler.extract_data(scene)
    #     print("DOne!")


    # if prepare_training_params["smooth"]:
    #     sampler = prepare_training_frames_hdf5.PrepareTrainingFramesHdf5(data_params_path,args[2],prep_toy,smooth = True)
    
    #     print("sampling frames smooth")
        
    #     for scene in scene_list:
    #         print(scene)
    #         sampler.extract_data(scene)
    #     print("DOne!")

    #     sampler = prepare_training_hdf5.PrepareTrainingHdf5(data_params_path,args[2],prep_toy,smooth = True)
        
    #     print("sampling trajectories smooth")
    #     for scene in scene_list:
    #         print(scene)

    #         sampler.extract_data(scene)
    #         print("DOne!")



    


    
    
    

if __name__ == "__main__":
    main()