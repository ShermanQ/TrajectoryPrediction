import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation



import json
import sys 
from learning.helpers.helpers_training import get_colors

class Animation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"
        self.reports_dir = self.data_params["reports_evaluation"] + "{}/".format(self.eval_params["report_name"])
        self.scene_samples = self.reports_dir + "{}_samples.json"
        self.gif_name = self.reports_dir + "{}_{}.gif"


        

    def animate_sample(self,scene,sample_id):
        file_ = json.load(open(self.scene_samples.format(scene)))
        sample = file_[str(sample_id)]
        inputs = np.array(sample["inputs"])
        labels = np.array(sample["labels"])
        outputs = np.array(sample["outputs"])



        prediction = np.concatenate([inputs,outputs], axis = 1)
        gt = np.concatenate([inputs,labels], axis = 1)

        nb_colors = gt.shape[0]

        colors = get_colors(nb_colors)

        animator = Animate(prediction,gt,colors,self.gif_name.format(scene,sample_id))
        animator.animate()

        

class Animate():
    def __init__(self,data_pred,data_gt,colors,gif_name = "test.gif", plot_ = False, save = True):
        self.xs_pred = data_pred[:,:,0]
        self.ys_pred = data_pred[:,:,1]

        self.xs_gt = data_gt[:,:,0]
        self.ys_gt = data_gt[:,:,1]


        self.nb_agents = self.xs_pred.shape[0]
        self.margin = 10

        self.nb_frames = self.xs_pred.shape[1]
        self.gif_name = gif_name
        self.plot_ = plot_
        self.save = save

        self.fps = 1
        self.colors = colors

        self.history = 4

        self.get_plots()



    def get_plots(self):
        self.fig, self.ax = plt.subplots(2,1,squeeze= False)
        self.plots1 = []
        self.plots2 = []


        for i in range(self.nb_agents):
            tup = (
                # self.ax[0][0].plot([], [], self.colors[i])[0],
                self.ax[0][0].plot([], [], self.colors[i],marker = 'o',markersize = 1.5,linewidth = 0.5)[0],
                # self.ax[0][0].plot([], [], self.colors[i],marker = 'x',markersize = 1.5,linewidth = 0.5)[0],


                # ,linestyle = ""
            )
            self.plots1.append(tup)

            tup = (
                # self.ax[1][0].plot([], [], self.colors[i])[0],
                self.ax[1][0].plot([], [], self.colors[i],marker = 'o',markersize = 1.5,linewidth = 0.5)[0]
            )
            self.plots2.append(tup)
        
            

    def animate(self):
        

        self.ax[0][0].set_xlim(np.min(self.xs_pred)-self.margin, np.max(self.xs_pred)+self.margin)
        self.ax[0][0].set_ylim(np.min(self.ys_pred)-self.margin, np.max(self.ys_pred)+self.margin)

        self.ax[1][0].set_xlim(np.min(self.xs_gt)-self.margin, np.max(self.xs_gt)+self.margin)
        self.ax[1][0].set_ylim(np.min(self.ys_gt)-self.margin, np.max(self.ys_gt)+self.margin)


        self.ax[1][0].set_title("groundtruth",loc = "right")
        self.ax[0][0].set_title("predictions",loc = "right")

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

        # if frame < 8:
        #     self.fig.suptitle('Observations', fontsize=16)
        # else:
        #     self.fig.suptitle('Predictions', fontsize=16)

        self.fig.suptitle("timestep: {}".format(frame+1), fontsize=8)

        
        for i,p in enumerate(self.plots1):

            # if frame < 8:
            p[0].set_data(self.xs_pred[i,start:end], self.ys_pred[i,start:end])
            p[0].set_color(self.colors[i])

            if frame > 7 :
                p[0].set_marker("+")
                p[0].set_markersize(2)

            
            #     p[1].set_data([], [])

            # else:
            #     p[0].set_data([], [])
            #     p[1].set_data(self.xs_pred[i,start:end], self.ys_pred[i,start:end])

            # p[1].set_data(self.xs_pred[i,:frame], self.ys_pred[i,:frame])

        for i,p in enumerate(self.plots2):
            p.set_data(self.xs_gt[i,start:end], self.ys_gt[i,start:end])
            p.set_color(self.colors[i])

            if frame > 7 :
                p.set_marker("+")
                p.set_markersize(2)

            # p[1].set_data(self.xs_gt[i,:frame], self.ys_gt[i,:frame])



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