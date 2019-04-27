
from evaluation.evaluation import Evaluation
import sys

# python model_evaluation.py parameters/data.json parameters/prepare_training.json
def main():
    args = sys.argv()

    eval = Evaluation(args[1],args[2])
    print("done!")