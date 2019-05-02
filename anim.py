import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animate():
    def __init__(self,data,scene = "",id = "", plot_ = True, save = True):
        self.xs = data[:,:,0]
        self.ys = data[:,:,1]
        self.nb_agents = self.xs.shape[0]
        self.margin = 10

        self.nb_frames = self.xs.shape[1]
        self.gif_name = "test.gif"
        self.plot_ = plot_
        self.save = save

        self.fps = 10
        self.colors = ["r","b"]

        self.get_plots()



    def get_plots(self):
        self.fig, self.ax = plt.subplots(2,1,squeeze= False)
        self.plots = []

        for i in range(self.nb_agents):
            tup = (
                self.ax[0][0].plot([], [], self.colors[i]+'o')[0],
                self.ax[0][0].plot([], [], self.colors[i])[0]
            )
            self.plots.append(tup)

    def animate(self):
        

        self.ax[0][0].set_xlim(np.min(self.xs)-self.margin, np.max(self.xs)+self.margin)
        self.ax[0][0].set_ylim(np.min(self.ys)-self.margin, np.max(self.ys)+self.margin)

        ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=self.nb_frames,repeat=True)

        if self.plot_:
            plt.show()
        if self.save:
            ani.save(self.gif_name, writer='imagemagick', fps=self.fps)



    def update(self,frame):
        frame = int(frame)
        for i,p in enumerate(self.plots):
            p[0].set_data(self.xs[i,:frame], self.ys[i,:frame])
            p[1].set_data(self.xs[i,:frame], self.ys[i,:frame])


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




    a = Animate(data)
    a.animate()


if __name__ == "__main__":
    main()





# x = np.array([0,1,2,3,3,3,3,3,3])
# y = np.array([0,0,0,0,0,1,2,3,4])

# x1 = np.array([-1,-1,-1,-1,-1,-1,-2,-3,-4])
# y1 = np.array([4,3,2,1,0,0,0,0,0])

# xs = np.concatenate([np.expand_dims(x,0),np.expand_dims(x1,0)],axis = 0)
# ys = np.concatenate([np.expand_dims(y,0),np.expand_dims(y1,0)],axis = 0)


# fig, ax = plt.subplots(2,1,squeeze= False)

# plots = []

# colors = ["r","b"]
# for i in range(xs.shape[0]):
#     tup = (
#         ax[0].plot([], [], colors[i]+'o')[0],
#         ax[0].plot([], [], colors[i])[0]
#     )
#     plots.append(tup)


# # ln, = plt.plot([], [], 'r')
# # ln1, = plt.plot([], [], 'ro')


# ax[0][0].set_xlim(np.min(xs)-10, np.max(xs)+10)
# ax[0][0].set_ylim(np.min(ys)-10, np.max(ys)+10)


# def update(frame):
#     frame = int(frame)
#     for i,p in enumerate(plots):
#         p[0].set_data(xs[i,:frame], ys[i,:frame])
#         p[1].set_data(xs[i,:frame], ys[i,:frame])





# ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(x),repeat=True)
# ani.save('animation.gif', writer='imagemagick', fps=10)
# plt.show()
