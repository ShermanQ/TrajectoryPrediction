import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from skimage import io,transform,util
import cv2


import json
import sys 
from learning.helpers.helpers_training import get_colors
from scipy.spatial.distance import euclidean

from scipy.misc import imread
import matplotlib.image as mpimg

class Animation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"
        self.reports_dir = self.data_params["reports_evaluation"] + "{}/".format(self.eval_params["report_name"])
        self.scene_samples = self.reports_dir + "{}_samples.json"
        self.gif_name = self.reports_dir + "{}_{}.gif"

        self.correspondences = json.load(open(self.data_params["sdd_pixel2meters"]))


        self.image = self.data_params["original_images"] + "{}.jpg"

        self.rev_dict_types = self.prepare_params["types_dic_rev"]
        


        

    def animate_sample(self,scene,sample_id):
        file_ = json.load(open(self.scene_samples.format(scene)))
        sample = file_[str(sample_id)]
        inputs = np.array(sample["inputs"])
        labels = np.array(sample["labels"])
        outputs = np.array(sample["outputs"])
        types = np.array(sample["types"])
        types = [ self.rev_dict_types[str(int(type_))] for type_ in types]

        self.__get_factor(scene)

        # img = cv2.imread(self.image.format(scene)).astype(int)

        # img = mpimg.imread(self.image.format(scene)).tolist()
        img = mpimg.imread(self.image.format(scene))



        # img = io.imread(self.image.format(scene)).tolist()
        # img = np.array(img,dtype = float)

        # img = [ [ [float(c) for c in b] for b in a] for a in img]

        

        print(type(img[0][0][0]))


        
        # io.imshow(img)
        # io.show()


        prediction = np.concatenate([inputs,outputs], axis = 1)
        gt = np.concatenate([inputs,labels], axis = 1)

        prediction = prediction * self.meter2pixel_ratio
        gt = gt * self.meter2pixel_ratio


        nb_colors = gt.shape[0]

        colors = get_colors(nb_colors)

        animator = Animate(prediction,gt,colors,img,types,self.gif_name.format(scene,sample_id))
        animator.animate()


    
    def __get_factor(self,scene):
        row = self.correspondences[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        self.pixel2meter_ratio = meter_dist/float(pixel_dist)
        self.meter2pixel_ratio = float(pixel_dist)/meter_dist


        

class Animate():
    def __init__(self,data_pred,data_gt,colors,img,types,gif_name = "test.gif", plot_ = False, save = True):

        self.img = img
        self.xs_pred = data_pred[:,:,0]
        self.ys_pred = data_pred[:,:,1]

        self.xs_gt = data_gt[:,:,0]
        self.ys_gt = data_gt[:,:,1]

        self.types = types 


        self.nb_agents = self.xs_pred.shape[0]
        self.margin = 1

        self.nb_frames = self.xs_pred.shape[1]
        self.gif_name = gif_name
        self.plot_ = plot_
        self.save = save

        self.fps = 1
        self.colors = colors

        self.lin_size = 100

        lin = np.linspace(0.6, 0.8, self.lin_size)
 
        self.color_dict = {
            "bicycle":cm.Blues(lin),
            "pedestrian":cm.Reds(lin),
            "car":cm.Greens(lin),
            "skate":cm.Greys(lin),
            "cart":cm.Purples(lin),
            "bus":cm.Oranges(lin)
        }

        self.colors = [self.color_dict[type_][np.random.randint(self.lin_size)] for type_ in self.types]

        self.history = 4

        self.get_plots()



    def get_plots(self):
        self.fig, self.ax = plt.subplots(1,2,squeeze= False)


        red_patch = mpatches.Patch(color='red', label='Pedestrians')
        blue_patch = mpatches.Patch(color='b', label='Bycicles')
        green_patch = mpatches.Patch(color='green', label='Cars')
        grey_patch = mpatches.Patch(color='grey', label='Skates')
        purple_patch = mpatches.Patch(color='purple', label='Carts')
        orange_patch = mpatches.Patch(color='orange', label='Buses')



        # red_patch = mpatches.Patch(color='red', label='Pedestrians')
        # red_patch = mpatches.Patch(color='red', label='Pedestrians')
        # red_patch = mpatches.Patch(color='red', label='Pedestrians')
        # red_patch = mpatches.Patch(color='red', label='Pedestrians')

        plt.legend(handles=[red_patch,blue_patch,green_patch,grey_patch,purple_patch,orange_patch],loc='best',fontsize = 3.5)


        self.ax[0][0].imshow(self.img,origin = "upper")
        self.ax[0][1].imshow(self.img,origin = "upper")
        # plt.show()


        self.plots1 = []
        self.plots2 = []


        for i in range(self.nb_agents):
            tup = self.ax[0][0].plot([], [], color = self.colors[i],marker = 'o',markersize = 2,linewidth = 0.5)[0]
            
                
            self.plots1.append(tup)

            tup = self.ax[0][1].plot([], [], color = self.colors[i],marker = 'o',markersize = 2,linewidth = 0.5)[0]
            
            self.plots2.append(tup)
        
            

    def animate(self):
        

        # self.ax[0][0].set_xlim(np.min(self.xs_pred)-self.margin, np.max(self.xs_pred)+self.margin)
        # self.ax[0][0].set_ylim(np.min(self.ys_pred)-self.margin, np.max(self.ys_pred)+self.margin)

        # self.ax[0][1].set_xlim(np.min(self.xs_gt)-self.margin, np.max(self.xs_gt)+self.margin)
        # self.ax[0][1].set_ylim(np.min(self.ys_gt)-self.margin, np.max(self.ys_gt)+self.margin)


        self.ax[0][1].set_title("Groundtruth",loc = "left", fontsize=8)
        self.ax[0][0].set_title("Predictions",loc = "left", fontsize=8)

        plt.tight_layout()

        ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=self.nb_frames,repeat=True)

        if self.plot_:
            plt.show()
        if self.save:
            ani.save(self.gif_name, writer='imagemagick', fps=self.fps,dpi = 200)



    def update(self,frame):
        frame = int(frame)
        end = frame + 1
        start = max(0,end-self.history)

        if end < 9:
            self.fig.suptitle("Timestep: {}, observation time".format(frame+1), fontsize=8)
        else:
            self.fig.suptitle("Timestep: {}, prediction time".format(frame+1), fontsize=8)
        
        for i,p in enumerate(self.plots1):

            p.set_data(self.xs_pred[i,start:end], self.ys_pred[i,start:end])
            # p.set_color(self.colors[i])

            if frame > 7 :
                p.set_marker("+")
                p.set_markersize(3)

                # p.set_fillstyle("none")


        for i,p in enumerate(self.plots2):
            p.set_data(self.xs_gt[i,start:end], self.ys_gt[i,start:end])
            # p.set_color(self.colors[i])

            if frame > 7 :
                p.set_marker("+")
                p.set_markersize(3)

    

# python evaluation/classes/sample_animations.py
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

    animate.animate_sample(eval_scenes[0],20)

    


if __name__ == "__main__":
    main()