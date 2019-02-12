import torch
import torch.nn as nn


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
        self.last_output = nn.Linear(self.hidden_size,self.input_size)

        

    def forward(self,x):
        self.hidden = self.init_hidden_state()
        output,self.hidden = self.lstm(x,self.hidden)
        decoder_start = self.last_output(output[:,-1])
        return decoder_start.view(self.batch_size,1,self.input_size), self.hidden

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
    
    def __init__(self,output_size,hidden_size,num_layers,batch_size):
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

        

    def forward(self,x,hidden):
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
    def __init__(self,input_size,output_size,hidden_size,hidden_size_d,num_layers,num_layers_d,batch_size,seq_len_d,teacher_forcing = False):
        super(seq2seq,self).__init__()
        self.encoder = encoderLSTM(input_size,hidden_size,num_layers,batch_size)
        self.decoder = decoderLSTM(output_size,hidden_size_d,num_layers_d,batch_size)
        self.seq_len_d = seq_len_d
        self.batch_size = batch_size
        self.output_size = output_size

    def forward(self,x):
        encoder_output, encoder_state = self.encoder(x)

    
        output,state = encoder_output,encoder_state
        # output = x[:,-1].view(200,1,2) # take last pos of input sequence as <sos>

        outputs = []
        for _ in range(self.seq_len_d):
            output,state = self.decoder(output,state)
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs.view(self.batch_size,self.seq_len_d,self.output_size)

