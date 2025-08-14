# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 22:54:13 2025

@author: eziol
"""

import sys
import os
import collections
import re
import torch
import math
import matplotlib.pyplot as plt
from torch import nn
import utility
import random
from torch.nn import functional as F
import matplotlib
matplotlib.use('TkAgg')
# 使用TkAgg来弹出一个交互窗口


# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 将当前目录加入 Python 路径


utility.DATA_HUB['time_machine'] = (utility.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():
    with open(utility.download('time_machine'),'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]
    # [^...]匹配非括号内的字符
    # A-Za-z表示所有的大小写字母
    # 
    
def tokenize(lines,token = 'word'):
    if token=='word':
        return [line.split() for line in lines]
    # str.split(sep=None, maxsplit=-1)
    # sep（可选）: 分隔符，默认为所有空白字符（如空格、换行、制表符等）。
    # maxsplit（可选）: 最大分割次数，默认为 -1（表示不限制次数）。
    
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：'+token)

class Vocab:
    def __init__(self,tokens = None,min_freq = 0,reserved_tokens = None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        counter = count_corpus(tokens)
        
        self._token_freqs = sorted(counter.items(),key = lambda x:x[1],
                                   reverse = True)
        # counter.items()输出元组列表，元组为(元素，出现次数)
        # reverse=True表示从高到低，降序排列；key = lambda x:x[1]，按元组的第二个元素排序
        self.idx_to_token = ['<unk>'] + reserved_tokens
        
        self.token_to_idx = {token:idx for idx,token in enumerate(self.idx_to_token)}
        # 将idx_to_token的（token）转化为{token:idx}的字典，其中token为字典的键，idx作为值
        
        for token,freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                # 将词添加进token_to_idx列表中，idx为其所在位置
                
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens, (list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self,indices):
        if not isinstance(indices, (list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    def unk(self):
        return 0
    
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens):
    if len(tokens) ==0 or isinstance(tokens[0], list):
        # 判断tokens是否存在，或者是否是list
        tokens = [token for line in tokens for token in line]
        # for line in tokens
        #     for token in line
        # 这个嵌套是这样的，将所有的词展平
    return collections.Counter(tokens)
    # collections.Counter返回一个类似字典的对象，Counter对象
    # {'apple':2,'banana':1}

def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = tokenize(lines,'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    
    if max_tokens > 0 :
        corpus = corpus[:max_tokens]
    return corpus,vocab

def seq_data_iter_random(corpus,batch_size,num_steps):
    # num_steps是每个子序列中预定义的时间步数（即每一个序列的长度）
    corpus = corpus[random.randint(0,num_steps - 1):]
    # randint给出从0到num_steps-1之间的任意整数，包含0和num_steps-1
    # -1是为了考虑标签
    num_subseqs = (len(corpus) - 1)//num_steps
    # 长度为num_steps的子序列起始索引
    initial_indices = list(range(0,num_subseqs * num_steps,num_steps))
    # 将原始的corpus分成多个子序列
    random.shuffle(initial_indices)
    # 将该序列进行打乱顺序，不一定在原始序列上相邻
    def data(pos):
        # 返回从pos位置开始，长度为num_step的序列
        return corpus[pos:pos + num_steps]
    
    num_batches = num_subseqs//batch_size
    for i in range(0,batch_size * num_batches,batch_size):
        # initial_indices为包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_sequential(corpus,batch_size,num_steps):
    # 生成顺序子序列
    offset = random.randint(0,num_steps)
    num_tokens = ((len(corpus)- offset - 1)//batch_size)*batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs,Ys = Xs.reshape(batch_size,-1),Ys.reshape(batch_size,-1)
    num_batches = Xs.shape[1]//num_steps
    for i in range(0,num_steps*num_batches,num_steps):
        X = Xs[:,i:i+num_steps]
        Y = Ys[:,i:i+num_steps]
        yield X, Y
        

class SeqDataLoader:
    # 加载序列数据的迭代器
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps = batch_size,num_steps
    def __iter__(self):
        # 在Python中，__iter__方法是实现迭代器协议的关键方法之一，用于使类的实例成为可迭代对象。以下是详细说明：
        # 定义__iter__方法后，对象可以通过for循环、next()函数、或生成器表达式等进行遍历。
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)

def load_data_time_machine(batch_size,num_steps,use_random_iter = False, max_tokens = 10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter,data_iter.vocab

def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device = device),)

def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size = shape, device = device) * 0.01
    W_xh = normal((num_inputs,num_hiddens))
    W_hh = normal((num_hiddens,num_hiddens))
    b_h = torch.zeros(num_hiddens,device = device)
    b_q = torch.zeros(num_outputs,device = device)

    params = [W_xh,W_hh,b_h,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def rnn(inputs,state,params):
    w_xh,w_hh,b_h,w_hq,b_q = params
    H, = state
    outputs = []
    # X的形状为：（批量大小，词表大小）
    for X in inputs:
        H = torch.tanh(torch.mm(X,w_xh) + H@w_hh)
        Y = H@w_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0),(H,)

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size,num_hiddens
        self.params = get_params(vocab_size,num_hiddens,device)
        self.init_state, self.forward_fn = init_state,forward_fn

    def __call__(self,x,state):
        # 类obj，创建一下，我们直接运行obj()，会调用obj.__call__
        x = F.one_hot(x.T,self.vocab_size).type(torch.float32)
        # 热编码，将一个有n个数据，h个类别的向量数据，把nx1大小的向量，转换成一个大小为
        # nxh的向量，例如[0,1,2],转换为[[1,0,0],[0,1,0],[0,0,1]]
        return self.forward_fn(x,state,self.params)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

def predict_ch8(prefix,num_preds,net,vocab,device):
    state = net.begin_state(batch_size = 1,device = device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]],device = device).reshape((1,1))
    # 函数名 = lambda 参数:表达式，等价于def函数名（参数）：return 表达式
    for y in prefix[1:]:
        _,state = net(get_input(),state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y,state = net(get_input(),state)
        outputs.append(int(y.argmax(dim = 1).reshape(1)))
        # argmax和argmin返回最大值或者最小值索引，dim指定维度，keepdim，是否保持维度
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:]*=theta/norm

def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
    state,timer = None, utility.Timer()
    metric = utility.Accumulator(2)
    for x,y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size = x.shape[0],device = device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
                # detach_()是原地操作，阻止某个张良成为计算图的一部分，避免计算梯度
            else:
                for s in state:
                    s.detach_()
        y = y.T.reshape(-1)
        x,y = x.to(device),y.to(device)
        y_hat, state = net(x,state)
        l = loss(y_hat,y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(batch_size = 1)
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1]),metric[1] / timer.stop()

def train_ch8(net,train_iter,vocab, lr, num_epochs,device, use_random_iter = False):
    loss = nn.CrossEntropyLoss()
    animator = utility.Animator(xlabel='epoch', ylabel = 'perplexity', legend = ['train'], xlim = [10,num_epochs])
    if isinstance(net,nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr = lr)
    else:
        updater = lambda batch_size: utility.sgd(net.params,lr,batch_size)
    predict = lambda prefix: predict_ch8(prefix,50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl,speed = train_epoch_ch8(net, train_iter,loss, updater, device, use_random_iter)
        if(epoch + 1)%10 ==0:
            print(predict('time traveller'))
            animator.add(epoch,[ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))
    print("Train completed！")
    plt.ioff()  # 关闭交互模式
    plt.show()

class RNNModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size,**kwargs):
        # **语法的作用是将后续接收到关键词参数打包成一个字典
        super(RNNModel,self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self,inputs,state):
        x = F.one_hot(inputs.T.long(),self.vocab_size)
        x = x.to(torch.float32)
        y,state = self.rnn(x,state)
        output = self.linear(y.reshape((-1,y.shape[-1])))
        return output,state
    def begin_state(self,device,batch_size):
        if not isinstance(self.rnn,nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),
                               device=device)
        else:
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


