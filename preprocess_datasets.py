from preprocess_datasets.stats import Stats
from preprocess_datasets.sdd_pixel2meter import Pixel2Meters
from preprocess_datasets.framerate_manager import FramerateManager
from preprocess_datasets.stops_remover import StopsRemover
from preprocess_datasets.data_augmenter import DataAugmenter
import sys
import json
import threading
import queue
import time

# def worker(manager,scene):
#     manager.change_rate(scene)

# def worker(q):
#     while True:
#         item = q.get()
#         manager = item[0]
#         scene = item[1]
#         manager.change_rate(scene)
#         q.task_done()

# python preprocess_datasets.py parameters/preprocessing.json parameters/data.json  parameters/prepare_training.json


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
    # q = queue.Queue()

    # num_worker = 2
    # for i in range(num_worker):
    #     t = threading.Thread(target=worker,args = (q,))
    #     t.daemon = True 
    #     t.start()

    s = time.time()

    print("Managing framerate")
    rate_manager = FramerateManager(args[2],new_rate)
    # threads = []
    for scene in scene_list:
        print(scene)
        # q.put((rate_manager,scene))
        # t = threading.Thread(target=worker,args = (rate_manager,scene,))
        # threads.append(t)
        rate_manager.change_rate(scene)
    

    # q.join()
    print("DOne!")

    print(time.time() - s)


    


    # ##################### conversion ###############
    pix2met = Pixel2Meters(args[2],1)
    for scene in scene_list:
        print(scene)
        pix2met.convert(scene)
    # ################################################

    


    # stats = Stats(args[1],args[2],args[3])
    # stats.get_stats()

    # print("removing full standing trajectories")
    # stops = StopsRemover(args[1],args[2],args[3])
    # for scene in scene_list:
    #     print(scene)
    #     stops.remove_stopped(scene)

    print("augmenting scenes")
    data_augmenter = DataAugmenter(args[1],args[2],args[3])
    for scene in scene_list:
        print(scene)
        data_augmenter.augment_scene(scene)
        data_augmenter.augment_scene_images(scene)

    


if __name__ == "__main__":
    main()