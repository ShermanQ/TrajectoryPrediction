import time
import helpers.helpers_data_torch as torch_data


"""
    the script master/prepare_training.py extracts data for training
    and store them into a csv file.
    This script creates for every sample two .pt files:
    One for the features and one for the labels and store them
    respectively in "./learning/data/samples/" and "./learning/data/labels/"
"""
def main():

    s = time.time()
    data_path = "./data/deep/data.csv"
    label_path = "./data/deep/labels.csv"

    samples_path = "./learning/data/samples/"
    labels_path = "./learning/data/labels/"
    
    # torch_data.extract_tensors(data_path,label_path,samples_path,labels_path)
    torch_data.extract_tensors_sophie(data_path,label_path,samples_path,labels_path)

    print(time.time()-s)
       
   

if __name__ == "__main__":
    main()




