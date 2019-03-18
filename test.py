import sys
import json
import csv 
import helpers
import numpy as np
# python test.py parameters/data.json parameters/prepare_training.json
def main():
    args = sys.argv
    data = json.load(open(args[1]))
    prepare = json.load(open(args[2]))

    scene_list = prepare["selected_scenes"]
    scene_path = data["preprocessed_datasets"] +"{}.csv"
    frames_temp = data["temp"] + "frames.txt"
    nb_agents_scenes = []
    for scene in scene_list:
        helpers.extract_frames(scene_path.format(scene),frames_temp,save = True)
        nb_agents_scene = []

        with open(frames_temp) as frames:
            for i,frame in enumerate(frames):
                # print(frame["ids"])
                frame = json.loads(frame)
                nb_agents = len(frame["ids"].keys())
                nb_agents_scene.append(nb_agents)
                nb_agents_scenes.append(nb_agents)

                # if nb_agents == 205:
                #     print(scene)
                #     print(i)

        print("{}: mean {}, std {}".format(scene,np.mean(nb_agents_scene),np.std(nb_agents_scene)))
        print(" max {}, min {}, nb_frames {}, nb_agents".format(np.max(nb_agents_scene),np.min(nb_agents_scene),i+1,np.sum(nb_agents_scene)))

        

        helpers.remove_file(frames_temp)
    print(" mean {}, std {}".format(np.mean(nb_agents_scenes),np.std(nb_agents_scenes)))
    print(" max {}, min {}".format(np.max(nb_agents_scenes),np.min(nb_agents_scenes)))


if __name__ == "__main__":
    main()