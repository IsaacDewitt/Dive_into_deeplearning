from pyexpat import features
from torch import nn
import math
import torch
import collections





class Encoder(nn.Module):
    def __init(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,x,**kwargs):
        raise NotImplementedError

class Decoder(nn.Module):
    def __intit(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    def init_state(self,enc_outputs, *args):
        raise NotImplementedError
    def forward(self,x,state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder, **kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.deocoder = decoder
    def foward(self,enc_x,dec_x,*args):
        # *args接受任意数量的参数，打包成一个元组
        # **kwargs接受任意数量的参数，打包成一个字典
        enc_outputs = self.encoder(enc_x,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_x,dec_state)
class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size,embed_size, num_hiddens, num_layers, dropout = 0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout = dropout)
        # Gated Recurrent Unit, GRU，门控循环单元,它和nn.LSTM结构相似
        # torch.nn.GRU(input_size, hidden_size, num_layers = 1, bias = True,batch_first = False,
        #              dropout = 0)
        # batch_first代表是否输出以(batch,seq,features)的形式
    def forward(self,X, *args):
        X = self.embedding(X)
        X = X.permute(1,0,2)
        output, state = self.rnn(X)
        # output的形状为(num_steps, batch_size,num_hiddens)
        # state的形状（num_layers,batch_size,num_hiddens）
        return output, state

# encoder = Seq2SeqEncoder(vocab_size = 10,embed_size = 8,num_hiddens = 16,num_layers = 2)
# encoder.eval()
# x = torch.zeros((4,7),dtype = torch.long)
# output, state = encoder(x)


