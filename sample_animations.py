
import json
import sys 
from evaluation.classes.animation import Animation



# python sample_animations.py
def main():
    args = sys.argv

    eval_params = json.load(open("parameters/model_evaluation.json"))
    data_params = json.load(open("parameters/data.json"))
    prepare_param = json.load(open("parameters/prepare_training.json"))

    # load scenes
    eval_scenes = prepare_param["eval_scenes"]
    train_eval_scenes = prepare_param["train_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]
    test_scenes = prepare_param["test_scenes"]

    # eval_ = Animation(args[1],args[2],args[3])
    animate = Animation("parameters/data.json","parameters/prepare_training.json","parameters/model_evaluation.json")

    animate.animate_sample(eval_scenes[0],100)

    


if __name__ == "__main__":
    main()