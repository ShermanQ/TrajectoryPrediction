from extractors import highd_extractor,koper_extractor,ngsim_extractor,sdd_extractor
from prepare_training import framerate_manager,digit_manager,scene_scaler,img_scaler,prepare_training,prepare_training_frames
import sys
import json


#python prepare_training parameters/parameters.json parameters/prepare_training.json parameters/data.json
def main():
    args = sys.argv

    parameters_path = json.load(open(args[1]))
    prepare_training_params = json.load(open(args[2]))
    data_params = args[3]

    # extractor_list = [
    #     highd_extractor.HighdExtractor(data_params,parameters_path["highd_extractor"]),
    #     koper_extractor.KoperExtractor(data_params,parameters_path["koper_extractor"]),
    #     ngsim_extractor.NgsimExtractor(data_params,parameters_path["ngsim_extractor"]),
    #     sdd_extractor.SddExtractor(data_params,parameters_path["sdd_extractor"])
    # ]

    # for extractor in extractor_list:
    #     extractor.extract()

    
    scene_list = prepare_training_params["selected_scenes"]
    # new_rate = float(prepare_training_params["framerate"])
    # rate_manager = framerate_manager.FramerateManager(args[3],new_rate)
    # for scene in scene_list:
    #     print(scene)
    #     rate_manager.change_rate(scene)

    # nb_digits = int(prepare_training_params["number_digits"])
    # digit_man = digit_manager.DigitManager(data_params,nb_digits)
    # for scene in scene_list:
    #     print(scene)
    #     digit_man.change_digit_number(scene)

    # center = int(prepare_training_params["center"])
    # scaler = scene_scaler.SceneScaler(data_params,center)
    # for scene in scene_list:
    #     print(scene)
    #     scaler.min_max_scale(scene)

    # img_size = int(prepare_training_params["img_size"])
    # scaler = img_scaler.ImgScaler(data_params,img_size)  

    # for scene in scene_list:
    #     print(scene)
    #     scaler.scale(scene)


    prepare_frames = int(prepare_training_params["samples_frames"])
    sampler = None
    if prepare_frames:
        print("frames sampling")
        sampler = prepare_training_frames.PrepareTrainingFrames(data_params,args[2])
    else:
        print("trajectory sampling")
        sampler = prepare_training.PrepareTraining(data_params,args[2])

    for scene in scene_list:
        print(scene)
        sampler.extract_data(scene)

if __name__ == "__main__":
    main()