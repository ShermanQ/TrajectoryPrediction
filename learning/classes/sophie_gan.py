import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp

def custom_mse(pred_seq,gt_seq):
    mse = nn.MSELoss(reduction= "none")

    # sum over a trajectory, average over batch size
    mse_loss = torch.mean(torch.sum(torch.sum(mse(pred_seq,gt_seq),dim = 2),dim = 1))

    return mse_loss

# class VOCfc32(nn.Module):
#     # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
#     def __init__(self,layers):
#         super(VOCfc32,self).__init__()

#         self.net = layers

#     def forward(self,x):
           
#         return self.net(x)
#input shape = (3*224*224)
class customCNN(nn.Module):
    
    def __init__(self,device,batch_size,nb_channels_in = 3, nb_channels_out = 512,nb_channels_projection = 128, input_size = 224, output_size = 7, embedding_size = 16,weights_path = "./learning/data/pretrained_models/voc_fc32_state.tar"):
        super(customCNN,self).__init__()

        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.nb_channels_in = nb_channels_in
        self.nb_channels_out = nb_channels_out
        self.weights_path = weights_path

        # out number of features is 25088 = 512 * 7 * 7 
        # self.cnn = torchvision.models.vgg19(pretrained=True).features
        self.__init_cnn()
        # main_model = imp.load_source('MainModel', "./learning/data/pretrained_models/vgg16_voc.py")
        # self.cnn = torch.load("./learning/data/pretrained_models/vgg16_voc.pth").to(device)
        # print(self.cnn)

        
        # self.embedding = nn.Linear(self.input_size,self.embedding_size)
        self.projection = nn.Conv2d(nb_channels_out,nb_channels_projection,1)
        self.nb_channels_projection = nb_channels_projection
        # self.embedding = nn.Linear(output_size**2, embedding_size)

        
    def __init_cnn(self):
        # print(torchvision.models.vgg16(pretrained=False))
        self.cnn = torchvision.models.vgg16(pretrained=False).features
        # print(self.cnn)
        
        self.cnn.load_state_dict(torch.load(self.weights_path)["state_dict"])
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn = self.cnn.to(self.device)

        

    def forward(self,x):
        x = x.view(self.batch_size,self.nb_channels_in,self.input_size,self.input_size)
        # print("test")
        cnn_features = self.cnn(x)
        projected_features = self.projection(cnn_features)
        # print("test {}".format(cnn_features.size()))

        # cnn_features = cnn_features.view(self.batch_size,-1)
        projected_features = projected_features.view(self.batch_size,self.nb_channels_projection,self.output_size**2) 
        projected_features = projected_features.permute(0,2,1)
        # projected_features = self.projection(cnn_features).view(self.batch_size,self.output_size**2) # B * 49

        # embedded_features = self.embedding(projected_features).view(self.batch_size,self.embedding_size)

        # return embedded_features
        return projected_features




"""
    Straightforward LSTM encoder

    returns hidden state at last time step
    and process last layer of hidden state through a linear layer
    to get its size to input_size
    this is a returned and will be used as starting point for decoding
"""
class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,batch_size,input_size = 2,hidden_size = 32,num_layers = 1, embedding_size = 16):
        super(encoderLSTM,self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.embedding = nn.Linear(self.input_size,self.embedding_size)

        self.lstm = nn.LSTM(input_size = self.embedding_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        

        

    # def forward(self,x,x_lengths,nb_max):
    def forward(self,x,x_lengths):

        hidden = self.init_hidden_state(len(x_lengths))
        x = self.embedding(x)
        x = f.relu(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x,hidden = self.lstm(x,hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        
        return x[:,-1,:], hidden[0].permute(1,2,0), hidden[1].permute(1,2,0)



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)





class SoftAttention(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,batch_size,input_size,output_size,hdec_size , nb_weights,apply_weigths_filter = True,layers = [64,128,64,1]):
        super(SoftAttention,self).__init__()
    
        # self.social_attention = SoftAttention(device,batch_size,enc_hidden_size,social_features_embedding_size,dec_hidden_size ,nb_neighbors_max)

        self.device = device
        self.hdec_size = hdec_size
        self.output_size = output_size
        self.input_size = input_size

        # self.nb_weights = nb_weights
        self.batch_size = batch_size
        self.apply_weigths_filter = apply_weigths_filter

        self.layers = [hdec_size + output_size] + layers
        self.features_embedding = nn.Linear(input_size,output_size)
        modules = []
        for i in range(1,len(self.layers)):
            modules.append(nn.Linear(self.layers[i-1],self.layers[i]))
            if i < len(self.layers) -1 :
                modules.append(nn.ReLU())
        self.core = nn.Sequential(*modules)
   

    def forward(self,hdec,features,zero_weigths = None):
        hdec = hdec.unsqueeze(1)
        hdec = hdec.repeat(1,features.size()[1],1)
        features = self.features_embedding(features)
        features = f.relu(features)

        inputs = torch.cat([features,hdec],dim = 2)

     
        attn = self.core(inputs)
        attn_weigths = f.softmax(attn.permute(0,2,1), dim = 2)

       

        attn_applied = torch.bmm(attn_weigths, features)
        return attn_applied


    
"""
    Straightforward LSTM decoder
    Uses prediction at timestep t-1 as input 
    to predict timestep t
"""
class decoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,batch_size,input_size,output_size,hidden_size,num_layers):
        super(decoderLSTM,self).__init__()

        self.device = device

        self.batch_size = batch_size
        self.input_size = input_size

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_size,num_layers = num_layers,batch_first = True)
        self.out = nn.Linear(self.hidden_size,self.output_size)

        

    def forward(self,x,hidden):
        # self.hidden = self.init_hidden_state()
        # x = x.view(self.seq_len,self.batch_size,2)
        output,self.hidden = self.lstm(x,hidden)
        output = self.out(output)#activation?
        return output, hidden

    def init_hidden_state(self):
        h_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)


class discriminatorLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,device,batch_size,input_size = 2,embedding_size = 16,hidden_size = 64,num_layers = 1,seq_len = 20):
        super(discriminatorLSTM,self).__init__()

        self.device = device

        self.batch_size = batch_size
        self.input_size = input_size
        self.embedding_size = embedding_size


        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_size = 1
        

        self.embedding = nn.Linear(input_size,embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size,hidden_size = hidden_size,num_layers = num_layers,batch_first = True)
        
        self.out = nn.Linear(self.hidden_size,self.output_size)

        

    def forward(self,x):
        self.hidden = self.init_hidden_state()

        # x = x.view(-1,self.input_size)
        x = self.embedding(x)
        # x = x.view(self.batch_size,self.seq_len,self.embedding_size)

        _,self.hidden = self.lstm(x,self.hidden)
        x = self.out(self.hidden[0])
        x = torch.sigmoid(x)
        return x

    def init_hidden_state(self):
        h_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,self.batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)
"""
    
"""
class sophie(nn.Module):
    def __init__(self,
            device,
            batch_size,
            enc_input_size = 2,
            enc_hidden_size = 32,
            enc_num_layers = 1, 
            embedding_size = 16,
            dec_hidden_size = 32,
            nb_neighbors_max = 204,
            social_features_embedding_size = 16,
            gaussian_dim = 8,
            dec_num_layer = 1,
            output_size = 2,
            pred_length = 12,
            obs_length = 8,
            # gpu_batch_limit = 12

            # disc_hidden_size = 64,
            # disc_nb_layer = 1

            ):
        super(sophie,self).__init__()
        self.device = device

        self.encoder = encoderLSTM(device,batch_size,enc_input_size,enc_hidden_size,enc_num_layers,embedding_size)
        self.batch_size = batch_size
        self.enc_input_size = enc_input_size
        self.enc_hidden_size = enc_hidden_size
        self.enc_num_layers = enc_num_layers
        self.embedding_size = embedding_size
        self.dec_hidden_size = dec_hidden_size
        self.nb_neighbors_max = nb_neighbors_max
        self.pred_length = pred_length
        self.obs_length = obs_length
        self.cnn = customCNN(device,batch_size)
        # batch_ size is either gpu_batch limit either a multiple of 
        # gpu_batch_limit divided in mini batches of size gpu_batch_limit
        # self.cnn = customCNN(device,gpu_batch_limit) 

        self.output_size = output_size
        # self.gpu_batch_limit = gpu_batch_limit

        # self.disc_hidden_size = disc_hidden_size
        # self.disc_nb_layer = disc_nb_layer



        self.gaussian = torch.distributions.MultivariateNormal(torch.zeros(gaussian_dim),torch.eye(gaussian_dim))
        self.social_features_embedding_size = social_features_embedding_size
        self.social_attention = SoftAttention(device,batch_size,enc_hidden_size,social_features_embedding_size,dec_hidden_size ,nb_neighbors_max,apply_weigths_filter = False)
        
        self.spatial_attention = SoftAttention(device,batch_size,128,embedding_size,dec_hidden_size ,embedding_size,apply_weigths_filter = False)
        
        # device,batch_size,input_size,output_size,hdec_size ,nb_weights,layers = [64,128,64])

        dec_input_size = gaussian_dim + 1*embedding_size + 1*embedding_size
        
        self.generator = decoderLSTM(device,batch_size,dec_input_size,output_size,dec_hidden_size,dec_num_layer)
        # self.discriminator = discriminatorLSTM(2*batch_size,enc_input_size,embedding_size,disc_hidden_size,disc_nb_layer,pred_length + obs_length)
        

    def forward(self,x,images,z):

        # get embedded spatial features
        # images = torch.randn(self.batch_size,3,224,224).to(self.device)
        Vsps = self.cnn(images)
        del images

        


        B,N,S,I = x.size()

        # number of active agent per batch
        agent_numbers = self.__get_number_agents(x.detach().cpu().numpy())
        
        # reshape so that new batch size is former batch_size * nb_max agents
        # output = x.view(B*N,S,I)

        # get lengths of sequences(without padding) in ascending order
        x_lengths = self.__get_lengths_x(x.cpu().detach().numpy())
        x_lengths = x_lengths.flatten()
        # x_lengths = np.array(x_lengths)

        
        
        

        # get the indices of the descending sorted lengths
        arg_ids = list(reversed(np.argsort(x_lengths)))

        # order input vector based on descending sequence lengths
        output = x.view(B*N,S,I)
        output = output[arg_ids]

        # get ordered/unpadded sequences lengths for pack_padded object
        sorted_x_lengths = x_lengths[arg_ids]

        # split the real agent and the padding agents
        padd_index = np.argmax(sorted_x_lengths == 0)
        output_nopad = output[:padd_index]
        sorted_x_lengths_nopad = sorted_x_lengths[:padd_index]

        # encode
        output_nopad,h,c = self.encoder(output_nopad,sorted_x_lengths_nopad)

        # concat with the padding agents
        pad_length = len(sorted_x_lengths) - len(sorted_x_lengths_nopad)
        pad_dims = [pad_length] + list(output_nopad.size()[1:])
        output_pad = torch.zeros(pad_dims)
        if torch.cuda.is_available():
            output_pad = output_pad.cuda()

        output = torch.cat([output_nopad,output_pad], dim = 0)

        pad_dims = [pad_length] + list(h.size()[1:])
        output_pad = torch.zeros(pad_dims)
        if torch.cuda.is_available():
            output_pad = output_pad.cuda()
        h = torch.cat([h,output_pad], dim = 0)
        c = torch.cat([c,output_pad], dim = 0)


        # reverse ordering of indices
        rev_arg_ids = np.argsort(arg_ids)
        # reverse ordering of encoded sequence
        output = output[rev_arg_ids]
        h = h[rev_arg_ids]
        c = c[rev_arg_ids]

        # reshape to original batch_size
        output = output.view(B,N,self.enc_hidden_size)
        h = h.view(B,N,self.enc_hidden_size,-1)
        c = c.view(B,N,self.enc_hidden_size,-1)


        
        # social features, sort hidden states by euclidean distance
        # remove main agent hidden state
        Vsos = self.__get_social_features(x_lengths,x,output)
       

        gen_init_hidden = (h[:,0].permute(2,0,1).contiguous(),c[:,0].permute(2,0,1).contiguous()) #temporary fix if lstm with more than 1 layer, it will fail
        generator_outputs = self.__generate(Vsos,Vsps,z,gen_init_hidden)


        return generator_outputs

    # def __generate(self,Vsos,Vsps,zero_weigths,z):
    def __generate(self,Vsos,Vsps,z,state):

        
        # state = self.generator.init_hidden_state()



        Csp = self.spatial_attention(state[0][0],Vsps)
        
        # Co = self.social_attention(state[0][0],Vsos,zero_weigths)
        Co = self.social_attention(state[0][0],Vsos)

        C = torch.cat([Co,Csp,z], dim = 2)

        outputs = []

        for _ in range(self.pred_length):
            
            output,state = self.generator(C,state)
            outputs.append(output)

            Csp = self.spatial_attention(state[0][0],Vsps)

            Co = self.social_attention(state[0][0],Vsos)
            C = torch.cat([Co,Csp,z], dim = 2)

        outputs = torch.stack(outputs).view(self.batch_size,self.pred_length,self.output_size)
        return outputs

    def __get_social_features(self,x_lengths,x,output):
        pad_dist = 1e10
        B,N,S,I = x.size()
        # get sequences last points
        last_points = torch.stack([s[i-1,:] for i,s in zip(x_lengths,x.view(B*N,S,I))])  # view really usefull
        last_points = last_points.view(B,N,I)

        # get main agent last point
        reference_points = last_points[:,0].view(B,1,self.enc_input_size).repeat(1,N,1)
       
        # compute euclidean distance
        dist = torch.sqrt(torch.sum(torch.pow(last_points-reference_points,2),dim = 2)).detach().cpu() # B*Nmax

        

        # set an infinite distance between padding agent and main agent
        x = [e[0] for e in  np.argwhere(x_lengths.reshape(B,N) == 0)]
        y = [e[1] for e in np.argwhere(x_lengths.reshape(B,N) == 0)]
        dist[x,y] = pad_dist

        Vsos = []
        for hiddens,d in zip(output,dist):              # est-ce que les padded agents sont toujours les plus éloignés? le faire de manière explicite -> non
            ids = np.argsort(d)                   # est-ce que l'attention peut se faire sans les padding agents --> taille variable du feature vector
            hiddens = hiddens[ids]
            Vso = hiddens-hiddens[0].repeat(N,1)

            # padding agent dummy value back to 0
            padding_agents = torch.zeros(self.enc_hidden_size)
            if torch.cuda.is_available():
                padding_agents = padding_agents.cuda()

            Vso[np.argwhere(d == pad_dist)] = padding_agents
            Vso = Vso[1:]
            Vsos.append(Vso)
        Vsos = torch.stack(Vsos).view(B,N-1,self.enc_hidden_size)

        return  Vsos

    # only works if padding is [0.,0.]
    def __get_number_agents(self,x):
        x = x.reshape(x.shape[0],x.shape[1],-1)
        x = np.sum(x,axis = 2)

        x = (x != 0.).astype(np.int)
        agent_numbers = np.sum(x,axis = 1)

       
        return agent_numbers
    # only works if padding is [0.,0.]

    def __get_lengths_x(self,x):
        x = np.sum(x,axis = 3)
        x = (x != 0.).astype(np.int)
        x_lengths = np.sum(x,axis = 2)
        return x_lengths










    # def __get_lengths_x(self,x,seq_len,padding = -1): ####################
    #     x_lengths = []
    #     for i,sequence in enumerate(x):
    #         unpadded_length = 0
    #         for point in sequence:
    #             if point[0] != padding:
    #                 unpadded_length += 1
    #         if unpadded_length == 0:
    #         #     empty_trajectories_ids.append(i)
    #             unpadded_length += 1

    #         x_lengths.append(unpadded_length)
    #     return x_lengths
    

    # def __get_number_agents(self,x,padding = 0.):
    #     agent_numbers = []
    #     for sample in x:
    #         nb_agent = 0
    #         for sequence in sample:
    #             # if sequence[-1][0] != padding:
    #             if sequence[0][0] != padding:                
    #                 nb_agent += 1
    #         agent_numbers.append(nb_agent)
    #     return agent_numbers

    



    # batch_size needs to be a multiple of gpu_batch_limit
    # def __gpu_memory_getaround(self,images):
    #     _,c,i,_ = images.size()
    #     if self.gpu_batch_limit + 1 > self.batch_size:
    #         return self.cnn(images.to(self.device))
    #     else:
    #         images = images.view(-1,self.gpu_batch_limit,c,i,i)
            

    #         features = []
    #         for img in images:
    #             print(img.size())
    #             features.append(self.cnn(img.to(self.device)).cpu())
    #             torch.cuda.synchronize()
                
    #         features = torch.stack(features).view(self.batch,c,i,i)
    #         return features



