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
import utility
import random



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