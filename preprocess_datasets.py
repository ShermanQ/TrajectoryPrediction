from preprocess_datasets.stats import Stats
from preprocess_datasets.sdd_pixel2meter import Pixel2Meters
from preprocess_datasets.framerate_manager import FramerateManager
import sys
import json

# python preprocess_datasets.py parameters/preprocessing.json parameters/data.json parameters/prepare_training.json 1
def main():
    args = sys.argv

    scene_list,train_list,test_list = None, None, None

    prepare_training_params = json.load(open(args[3]))


    prep_toy = prepare_training_params["toy"]

    if prep_toy:
        scene_list = prepare_training_params["toy_scenes"]
        train_list = prepare_training_params["toy_train_scenes"]
        test_list = prepare_training_params["toy_test_scenes"]
    else:
        scene_list = prepare_training_params["selected_scenes"]
        train_list = prepare_training_params["train_scenes"]
        test_list = prepare_training_params["test_scenes"]

    new_rate = float(prepare_training_params["framerate"])

    #################framerate#######################

    # print("Managing framerate")
    # rate_manager = FramerateManager(args[2],new_rate)
    # for scene in scene_list:
    #     print(scene)
    #     rate_manager.change_rate(scene)
    # print("DOne!")

    ##################### conversion ###############
    # pix2met = Pixel2Meters(args[2],args[4])
    # for scene in scene_list:
    #     print(scene)
    #     pix2met.convert(scene)
    ################################################

    


    stats = Stats(args[1],args[2])
    stats.get_stats()

    
    


if __name__ == "__main__":
    main()