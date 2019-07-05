from prepare_training import prepare_training_hdf5,prepare_training_frames_hdf5
from prepare_training import img_scaler

import sys
import json
import csv
import helpers
import h5py
import os
import time

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
    else:
        scene_list = prepare_training_params["selected_scenes"]
        test_list = prepare_training_params["test_scenes"]

        # scene_list = helpers.augment_scene_list(scene_list,preprocessing["augmentation_angles"])
        # train_list = helpers.augment_scene_list(train_list,preprocessing["augmentation_angles"])
        # test_list = helpers.augment_scene_list(test_list,preprocessing["augmentation_angles"])

    for scene in scene_list:
        os.system(" rm  {}{}.csv".format(data["preprocessed_datasets"],scene))
        os.system(" cp {}{}.csv {}{}.csv".format(data["filtered_datasets"],scene,data["preprocessed_datasets"],scene))

 
    s = time.time()


    # img_size = int(prepare_training_params["img_size"])
    # scaler = img_scaler.ImgScaler(data_params_path,img_size)  
    # print("scaling images")
    # for scene in helpers.augment_scene_list(scene_list,preprocessing["augmentation_angles"]):
    #     print(scene)
    #     scaler.scale(scene)


   
    print(time.time()-s)
    sampler = prepare_training_hdf5.PrepareTrainingHdf5(data_params_path,args[2],prep_toy,smooth = False)

    # For the test scenes, we increase the shift value and set it to the duration
    # of the observation duration
    print("sampling trajectories")
    for scene in scene_list:
        print(scene)
        if scene in test_list:
            print("scene in test: changing shift value")
            shift_temp = sampler.shift
            sampler.shift = prepare_training_params["t_pred"]
            sampler.extract_data(scene)
            sampler.shift = shift_temp
        else:
            sampler.extract_data(scene)
        print("DOne!")

    print(time.time()-s)

   
    if prepare_training_params["smooth"]:
        sampler = prepare_training_hdf5.PrepareTrainingHdf5(data_params_path,args[2],prep_toy,smooth = True)
        
        print("sampling trajectories smooth")
        for scene in scene_list:
            print(scene)

            sampler.extract_data(scene)
            print("DOne!")

            print(time.time()-s)



 # sampler = prepare_training_frames_hdf5.PrepareTrainingFramesHdf5(data_params_path,args[2],prep_toy,smooth = False)
    
    # print("sampling frames")
    
    # for scene in scene_list:
    #     print(scene)
    #     sampler.extract_data(scene)
    # print("DOne!")

    
 # if prepare_training_params["smooth"]:
    #     sampler = prepare_training_frames_hdf5.PrepareTrainingFramesHdf5(data_params_path,args[2],prep_toy,smooth = True)
    
    #     print("sampling frames smooth")
        
    #     for scene in scene_list:
    #         print(scene)
    #         sampler.extract_data(scene)
    #     print("DOne!")

    #     print(time.time()-s)


    
    
    

if __name__ == "__main__":
    main()