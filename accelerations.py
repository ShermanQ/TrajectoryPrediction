from evaluation.classes.acceleration_distribution import Accelerations
import sys
import json
# python accelerations.py parameters/data.json parameters/prepare_training.json parameters/model_evaluation.json 

def main():
    args = sys.argv
    print(args)

    # data = args[0]
    # prepare = args[1]
    # parameters = args[2]

    data = "parameters/data.json"
    prepare = "parameters/prepare_training.json"
    eval = "parameters/model_evaluation.json"



    # eval_params = json.load(open("parameters/model_evaluation.json"))
    # data_params = json.load(open("parameters/data.json"))
    # prepare_param = json.load(open("parameters/prepare_training.json"))

    acc = Accelerations(data,prepare,eval)
    acc.get_distribs()

   


if __name__ == "__main__":
    main()