import torch
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_params(named_parameters,epoch=0,root="./data/reports/weights/"):
    weights = {}
    for n,p in named_parameters:
        if(p.requires_grad) :
            weights[n] = p.cpu().detach().numpy().flatten()

    n_rows = int(np.ceil( np.sqrt(len(weights))) )


    fig,axs = plt.subplots(n_rows-1,n_rows,sharex=False,sharey=False,squeeze = False)
    ctr = 0
    fig.set_figheight(15)
    fig.set_figwidth(30)

    layers = list(weights.keys())
    
    for i in range(n_rows-1 ):
        for j in range(n_rows):

            if ctr < len(weights) :
                axs[i][j].hist(weights[layers[ctr]],label = layers[ctr],bins = 20)
                # axs[i][j].set_title("weights {}".format(layers[ctr]))
                lgd = axs[i][j].legend()
            ctr += 1
    fig.tight_layout()
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

path = "./learning/data/models/model_20_1557345347.4582756.tar"
state = torch.load(path)
args= state["args"]
state_dict = state["state_dict"]

# plot_params(state_dict)

types_dic_rev={
        1:"bicycle",
        2:"pedestrian",
        3:"car",
        4:"skate",
        5:"cart",
        6:"bus"
    }

nb_cats = args["nb_cat"]
embedding_size = args["word_embedding_size"]

weights = state_dict["type_embedding.weight"]

# plt.hist(torch.cuda.FloatTensor(weights).detach().cpu().numpy() ,bins = 20)
# plt.show()

print(type(weights))

embedding_trained = torch.nn.Embedding(nb_cats,embedding_size)
# plt.hist(embedding_trained.weight.detach().cpu().numpy(),bins = 20 )
# plt.show()

embedding_trained.weight = torch.nn.Parameter(torch.cuda.FloatTensor(weights), requires_grad=False)
# plt.hist(embedding_trained.weight.detach().cpu().numpy(),bins = 20 )
# plt.show()

embeding_random = torch.nn.Embedding(nb_cats,embedding_size)

# plt.hist(embeding_random.weight.detach().cpu().numpy(),bins = 20 )
# plt.show()
keys = torch.LongTensor(np.arange(0,6))
labels = [types_dic_rev[key.item()+1] for key in keys]

embeding_random = embeding_random.to(device)
embedding_trained = embedding_trained.to(device)
keys = keys.to(device)






t = embedding_trained(keys)
r = embeding_random(keys)

t = t.detach().cpu().numpy()
r = r.detach().cpu().numpy()


# t = TSNE(n_components=2).fit_transform(t)
# r = TSNE(n_components=2).fit_transform(r)

t = PCA(n_components=2).fit_transform(t)
r = PCA(n_components=2).fit_transform(r)

fig,axs = plt.subplots(1,2,squeeze=False)
print(t[0][0],t[0][1])
axs[0][0].set_title("Trained")
axs[0][1].set_title("Random")

for i,key in enumerate(labels):
    axs[0][0].scatter([t[i][0]],[t[i][1]],label = key)
    axs[0][1].scatter([r[i][0]],[r[i][1]],label = key)
plt.legend()
plt.show()




