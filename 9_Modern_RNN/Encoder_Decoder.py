from pyexpat import features
from torch import nn
import math
import torch
import collections

from torch.utils.hipify.hipify_python import meta_data

from utility import Animator, Timer, Accumulator, try_gpu, load_data_nmt
import utility


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
        self.decoder = decoder
    def forward(self,enc_x,dec_x,*args):
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
        X = self.embedding(X).permute(1,0,2)
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
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred,label, valid_len):
        weights = torch.ones_like(label)
        #返回一个和input大小相同的全是1的张量
        # 这一段很傻逼的是valide_len这个东西，一直没有给出来
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0,2,1), label)
        weigted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weigted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch',ylabel='loss',xlim = [10,num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            x,x_valid_len,y,y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * y.shape[0],device=device).reshape(-1,1)
            dec_input = torch.cat([bos,y[:,:-1]],1)
            y_hat,_ = net(x,dec_input,x_valid_len)
            l = loss(y_hat,y,y_valid_len)
            l.sum().backward()
            utility.grad_clipping(net,1)
            num_tokens = y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(),num_tokens)
        if (epoch+1)%10==0:
            animator.add(epoch+1,(metric[0]/metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f},{metric[1]/timer.stop():.1f}'
          f'tokens/sec on {str(device)}')
if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32,32,2,0.1
    batch_size, num_steps = 64,10
    lr, num_epochs, device = 0.005,300,try_gpu()
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size,num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
    net = EncoderDecoder(encoder,decoder)
    train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
