import torch 
import imp 
import numpy as np 
import torchvision
import copy

path = "./learning/data/pretrained_models/"
device = torch.device("cuda:0")
# load model converted
MainModel = imp.load_source('MainModel', path+"vgg16_voc.py") 
model1 = torch.load(path+"vgg16_voc.pth")
## state_dict
imported_state = model1.state_dict()

# load native model pytorch
model2 = torchvision.models.vgg16(pretrained=True).features
## state_dict
native_state = model2.state_dict()


imported_state_copy = copy.deepcopy(imported_state)
for key,new_key in zip(imported_state.keys(),native_state.keys()):
	imported_state_copy[new_key] = imported_state_copy.pop(key)

keys_to_drop = ['fc6.weight', 'fc6.bias', 'fc7.weight', 'fc7.bias', 'score_fr.weight', 'score_fr.bias']

for key in keys_to_drop:
	imported_state_copy.pop(key)

the_model = torchvision.models.vgg16(pretrained=False).features
the_model.load_state_dict(imported_state_copy)

state = {
        'state_dict': the_model.state_dict(),
        }

torch.save(state, path+"voc_fc32_state.tar")
#for param in the_model.parameters():
    #param.requires_grad = False

#the_model = the_model.to(device)


