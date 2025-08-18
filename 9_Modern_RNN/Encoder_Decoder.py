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
class Seq2SeqDecoder(Decoder):
    def __init__(self,vocab_size,embed_size, num_hiddens, num_layers, dropout = 0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout = dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self,enc_outputs, *args):
        return enc_outputs[1]
    def forward(self,X, state):
        # 说实话感觉作者不是特别懂这一块啊,整个这一章有点太,那啥了
        # 输出大小为(batch_size, num_steps, embed_size)
        X = self.embeding(X).permute(1,0,2)
        # 将tensor的维度重新排列,默认是(0,1,2),这里排列成(1,0,2)
        context = state[-1].repeat(X.shape[0],1,1)
        # repeat函数表示state最后一个状态在某个方向复制多少次,这里表示在行方向重复X.shape[0]次,其他方向上不重复
        X_and_context = torch.cat((X,context),2)
        output,state = self.rnn(X_and_context,state)
        output = self.dense(output).permute(1,0,2)
        # output形状变为（batch_size, numsteps,vocab_size)
        # state的形状为：（num_layers,batch_size,num_hiddens)
        return output,state

def sequence_mask(X, valid_len,value = 0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen,dtype = torch.float32, device = X.device)[None,:]<valid_len[:,None]
    # torch.arange(start=0, end, step=1, *, dtype=None, device=None)
    # 行为类似range，在这里的None的行为是在张量中插入一个新维度，这里防止mask变成(maxlen,)，使其成为(1, maxlen)
    # 这里利用了广播特性，<左边的大小为[1,maxlen],右边为valid_len[batch_size,1],然后两个都会变成
    # [batch_size,maxlen],进行比较


# encoder = Seq2SeqEncoder(vocab_size = 10,embed_size = 8,num_hiddens = 16,num_layers = 2)
# encoder.eval()
# x = torch.zeros((4,7),dtype = torch.long)
# output, state = encoder(x)


