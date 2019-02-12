import time
import helpers.helpers_data_torch as torch_data

def main():

    s = time.time()
    data_path = "./data/deep/data.csv"
    label_path = "./data/deep/labels.csv"

    samples_path = "./learning/data/samples/"
    labels_path = "./learning/data/labels/"
    
    torch_data.extract_tensors(data_path,label_path,samples_path,labels_path)
    print(time.time()-s)
       
   

if __name__ == "__main__":
    main()




