import json
import sys 


class Animation():
    def __init__(self,data_params,prepare_params,eval_params):
        self.data_params = json.load(open(data_params))
        self.prepare_params = json.load(open(prepare_params))
        self.eval_params = json.load(open(eval_params))


        self.models_path = self.data_params["models_evaluation"] + "{}.tar"
        self.reports_dir = self.data_params["reports_evaluation"] + "{}/".format(self.eval_params["report_name"])
        self.scene_samples = self.reports_dir + "{}_samples.json"

        

    def animate_sample(self,scene,sample_id):
        file_ = json.load(open(self.scene_samples.format(scene)))
        sample = file_[str(sample_id)]
        inputs = sample["inputs"]
        labels = sample["labels"]
        outputs = sample["outputs"]

        prediction = inputs + outputs
        gt = inputs + labels

        fig,axs = plt.subplots(2,1,sharex=True,sharey=True,squeeze = False)

        colors = np.array(get_colors(len(kept_samples[plot][0])))

        if len(colors) > 0:          
            last_points = []
            r = 0
            c = 0
            for j,agent in enumerate(kept_samples[plot][0]):
                               

                color = colors[j]
                agent = agent.reshape(-1,2)

                
                agent = [e for e in agent if e[0]!=0 and e[1] != 0]
                

                if len(agent) > 0:

            
                    x = [e[0] for e in agent]
                    y = [e[1] for e in agent]
                    
                    axs[r][c].plot(x,y,color = color)

                    if j == 0:
                        axs[r][c].scatter(x,y,marker = "+",color = color,label = "obs")
                        axs[r][c].scatter(x[0],y[0],marker = ",",color = color,label = "obs_start")
                        axs[r][c].scatter(x[-1],y[-1],marker = "o",color = color,label = "obs_end")
                    else :

                        axs[r][c].scatter(x,y,marker = "+",color = color)
                        axs[r][c].scatter(x[0],y[0],marker = ",",color = color)
                        axs[r][c].scatter(x[-1],y[-1],marker = "o",color = color)
        




# python evaluation/classes/animations.py
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

    animate.animate_sample(eval_scenes[0],1)

    


if __name__ == "__main__":
    main()