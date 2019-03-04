import time
import helpers.helpers_data_torch as torch_data
import json 
import sys 

"""
    the script master/prepare_training.py extracts data for training
    and store them into a csv file.
    This script creates for every sample two .pt files:
    One for the features and one for the labels and store them
    respectively in "./learning/data/samples/" and "./learning/data/labels/"
"""
# python learning/data_torch.py parameters/data.json parameters/prepare_training.json
def main():
    args = sys.argv
    data = json.load(open(args[1]))
    param = json.load(open(args[2]))

    s = time.time()
    
    data_path = data["prepared_samples_grouped"]
    label_path = data["prepared_labels_grouped"]
    scenes = json.load(open(data["prepared_ids"]))["ids"]

    images_path = data["prepared_images"]

    samples_path = "./learning/data/samples/"
    labels_path = "./learning/data/labels/"
    img_path = "./learning/data/img/"
    
    # torch_data.extract_tensors(data_path,label_path,samples_path,labels_path)
    torch_data.extract_tensors_sophie(data_path,label_path,scenes,samples_path,labels_path,img_path,images_path)

    print(time.time()-s)
       
   

if __name__ == "__main__":
    main()




