from learning.classes.pretrained_vgg import customCNN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# device = torch.device("cpu")   
import cv2
import numpy as np
import skimage.io as io

from learning.classes.fcn32s import FCN32s
import matplotlib.pyplot as plt

from collections import OrderedDict



img_path = "./data/prepared_datasets/images/bookstore0.jpg"
img = np.array(cv2.imread(img_path))
print(img.shape)

img = torch.FloatTensor(img).to(device)
img = img.permute(2,0,1).unsqueeze(0)
print(img.size())


c = FCN32s()
pretrained_dict = torch.load("./learning/data/pretrained_models/fcn32s_from_caffe.pth")
new_dict = OrderedDict()
for k,v in pretrained_dict.items():
    if k in c.state_dict():
        new_dict[k] = v
c.load_state_dict(new_dict)

c.to(device)
out = c(img)

# out = out.squeeze(0).permute(1,2,0).detach().cpu().numpy()

# a = np.argmax(out,axis = -1)




