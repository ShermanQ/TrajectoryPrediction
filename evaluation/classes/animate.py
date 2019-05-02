
class Animate():
    def __init__(self,data_pred,data_gt,colors,gif_name = "test.gif", plot_ = True, save = True):
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

        self.fps = 10
        self.colors = colors

        self.get_plots()



    def get_plots(self):
        self.fig, self.ax = plt.subplots(2,1,squeeze= False)
        self.plots1 = []
        self.plots2 = []


        for i in range(self.nb_agents):
            tup = (
                self.ax[0][0].plot([], [], self.colors[i]+'o')[0],
                self.ax[0][0].plot([], [], self.colors[i])[0]
            )
            self.plots1.append(tup)

            tup = (
                self.ax[1][0].plot([], [], self.colors[i]+'o')[0],
                self.ax[1][0].plot([], [], self.colors[i])[0]
            )
            self.plots2.append(tup)
        
            

    def animate(self):
        

        self.ax[0][0].set_xlim(np.min(self.xs_pred)-self.margin, np.max(self.xs_pred)+self.margin)
        self.ax[0][0].set_ylim(np.min(self.ys_pred)-self.margin, np.max(self.ys_pred)+self.margin)

        self.ax[1][0].set_xlim(np.min(self.xs_gt)-self.margin, np.max(self.xs_gt)+self.margin)
        self.ax[1][0].set_ylim(np.min(self.ys_gt)-self.margin, np.max(self.ys_gt)+self.margin)

        ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=self.nb_frames,repeat=True)

        if self.plot_:
            plt.show()
        if self.save:
            ani.save(self.gif_name, writer='imagemagick', fps=self.fps)



    def update(self,frame):
        frame = int(frame)
        for i,p in enumerate(self.plots1):
            p[0].set_data(self.xs_pred[i,:frame], self.ys_pred[i,:frame])
            p[1].set_data(self.xs_pred[i,:frame], self.ys_pred[i,:frame])

        for i,p in enumerate(self.plots2):
            p[0].set_data(self.xs_gt[i,:frame], self.ys_gt[i,:frame])
            p[1].set_data(self.xs_gt[i,:frame], self.ys_gt[i,:frame])


def main():

    x = np.array([0,1,2,3,3,3,3,3,3])
    y = np.array([0,0,0,0,0,1,2,3,4])

    x1 = np.array([-1,-1,-1,-1,-1,-1,-2,-3,-4])
    y1 = np.array([4,3,2,1,0,0,0,0,0])

    xs = np.concatenate([np.expand_dims(x,0),np.expand_dims(x1,0)],axis = 0)
    ys = np.concatenate([np.expand_dims(y,0),np.expand_dims(y1,0)],axis = 0)

    xs = np.expand_dims(xs,2)
    ys = np.expand_dims(ys,2)

    data = np.concatenate([xs,ys],-1)




    a = Animate(data,data)
    a.animate()


if __name__ == "__main__":
    main()


