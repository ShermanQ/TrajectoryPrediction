import torch
import torch.nn as nn
import torch.nn.functional as f 
 

"""
    Straightforward LSTM encoder

    returns hidden state at last time step
    and process last layer of hidden state through a linear layer
    to get its size to input_size
    this is a returned and will be used as starting point for decoding
"""
class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,input_size,hidden_size,num_layers,batch_size):
        super(encoderLSTM,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.output_size = output_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden_state()

        self.lstm = nn.LSTM(input_size = self.input_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        # self.last_output = nn.Linear(self.hidden_size,self.input_size)

        

    def forward(self,x):
        self.hidden = self.init_hidden_state()
        output,self.hidden = self.lstm(x,self.hidden)
        # decoder_start = self.last_output(output[:,-1])
        # return decoder_start.view(self.batch_size,1,self.input_size), self.hidden
        # print(type(self.hidden),len(self.hidden))
        # print(self.hidden[0].size())
        # print("in")

        return output, self.hidden


    def init_hidden_state(self):
        h_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()

        return (h_0,c_0)


"""
    Straightforward LSTM decoder
    Uses prediction at timestep t-1 as input 
    to predict timestep t
"""
class decoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,output_size,hidden_size,num_layers,batch_size,seq_len):
        super(decoderLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden_state()
        # self.seq_len = seq_len
        # self.target = target

        
        self.lstm = nn.LSTM(input_size = self.output_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        self.out = nn.Linear(self.hidden_size,self.output_size)

        

    def forward(self,x,hidden,encoder_outputs):
        # self.hidden = self.init_hidden_state()
        # x = x.view(self.seq_len,self.batch_size,2)
        output,self.hidden = self.lstm(x,hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden_state(self):
        h_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()

        return (h_0,c_0)


"""
    Uses encoderLSTM and decoderLSTM in order to achieve training
    inputs goes through encoderLSTM which outputs hidden state at 
    last time step and start input for decoder.
    Then uses decodeLSTM, seq_len_d times to make the predictions
    It uses predictions of previous timestep to predict for the current timestep
    THe decoder outputs are stored and returned
"""
class seq2seq(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,hidden_size_d,num_layers,num_layers_d,batch_size,seq_len,seq_len_d,attention = False):
        super(seq2seq,self).__init__()
        self.encoder = encoderLSTM(input_size,hidden_size,num_layers,batch_size)
        # 
        if attention:
            print("Attention enabled")
            self.decoder = decoderAttentionLSTM(output_size,hidden_size_d,num_layers_d,batch_size,seq_len)
        else: 
            print("Attention disabled")
            self.decoder = decoderLSTM(output_size,hidden_size_d,num_layers_d,batch_size,seq_len)
        # self.attention = attention

        self.seq_len_d = seq_len_d
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.output_size = output_size

    def forward(self,x):
        encoder_output, encoder_state = self.encoder(x)

    
        # output,state = encoder_output,encoder_state
        state = encoder_state

        output = x[:,-1].view(self.batch_size,1,self.output_size) # take last pos of input sequence as <sos>
        

        outputs = []
        for _ in range(self.seq_len_d):
            output,state = self.decoder(output,state,encoder_output)
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs.view(self.batch_size,self.seq_len_d,self.output_size)


class decoderAttentionLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):
    
    def __init__(self,output_size,hidden_size,num_layers,batch_size,seq_len):
        super(decoderAttentionLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden_state()
        self.seq_len = seq_len
        # self.target = target

        self.attn = nn.Linear(self.hidden_size*2  + self.output_size,self.seq_len)#8
        # self.attn_c = nn.Linear(self.hidden_size  + self.output_size,self.seq_len)#8


        self.attn_combine = nn.Linear(self.hidden_size + self.output_size,self.output_size)
        self.lstm = nn.LSTM(input_size = self.output_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)
        self.out = nn.Linear(self.hidden_size,self.output_size)

        

    def forward(self,x,hidden,encoder_outputs):
        # self.hidden = self.init_hidden_state()
        # x = x.view(self.seq_len,self.batch_size,2)
        

        # attention weights
        last_state = hidden[0][-1].view(self.batch_size,1,self.hidden_size)
        last_cell = hidden[1][-1].view(self.batch_size,1,self.hidden_size)
        input_last_state = torch.cat((x,last_state,last_cell),dim = 2).view(self.batch_size,1,self.hidden_size * 2 +self.output_size)
        # input_last_cell = torch.cat((x,last_cell),dim = 2).view(self.batch_size,1,self.hidden_size +self.output_size)

        attn_weights_state = f.softmax(self.attn(input_last_state),dim = 2)
        # attn_weights_cell = f.softmax(self.attn_c(input_last_cell),dim = 2)

        
        attn_applied = torch.bmm(attn_weights_state, encoder_outputs)

        attn_combined = self.attn_combine(torch.cat((x,attn_applied),dim = 2))

        x = f.relu(attn_combined)
        

        output,self.hidden = self.lstm(x,hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden_state(self):
        h_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size).cuda()

        return (h_0,c_0)

