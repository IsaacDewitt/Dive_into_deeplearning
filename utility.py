# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:59:32 2025

@author: eziol
"""
import torchvision
from torchvision import transforms
import numpy as np
import torch
import time
import hashlib
from torch.utils import data
import matplotlib.pyplot as plt
import os
import requests
import tarfile
import zipfile
import sys
import re
import random
import collections
from torch.nn import functional as F
from torch import nn
import math
# requests是python中常用的HTTP请求库之一，可以实现python中的发送网络请求和接收网路请求
import matplotlib

matplotlib.use('TkAgg')
# 使用TkAgg来弹出一个交互窗口
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
def download(name,cache_dir = os.path.join('..','data')):
    DATA_HUB = dict()
    # 创建字典
    DATA_HUB['kaggle_house_train'] = (  # @save
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (  # @save
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url,sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir,exist_ok = True)
    # exist_ok为True代表目录如果存在也不报错，该函数是为了创建一个目录
    fname = os.path.join(cache_dir,url.split('/')[-1])
    # os.path.join(path1, path2, ..., pathN)，将多个path拼接在一起，并且使用正确的分隔符
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        # 创建一个SHA-1哈希算法的对象，初始化了一个SHA-1哈希对象
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            # 缓存命中
            return fname
            
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url,stream = True, verify = True)
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname



class Accumulator:
    def __init__(self,n):
        self.data = [0.0] *n
        # 这种写法是创建一个长度为n的列表
        
    def add(self,*args):
        # *args代表可变参数，所有传入的参数会被合并为一个名为args的元组中
        self.data = [a+float(b)for a,b in zip(self.data,args)]
        
    def reset(self):
        self.data = [0.0] *len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def synthetic_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    # noraml用来产生符合正态（高斯）分布的随机数，第一个值是mean，默认为0，第二个是标准差，默认为1，第三个为size，
    # 类型为元组，如果是一个整数，则产生一个一维张量
    y = torch.matmul(x,w) + b
    # matmul是来执行矩阵乘法的函数，接受两个输入，两个输入进行矩阵乘法
    y+=torch.normal(0,0.001,y.shape)
    # 添加噪声
    return x, y.reshape((-1,1))
    # 将y整形成一个二维矩阵（x*1）的形式

def evaluate_loss(net,data_iter,loss):
    metric = Accumulator(2)
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]


def load_data(data_arrays,batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    # PyTorch中的一个用于快速封装张量数据的工具类，将张量封装成一个统一的Dataset对象
    # 定义为class torch.utils.data.TensorDataset(*tensors)，传入的张量的第一个维度必须一致
    # 传入的张量第一个维度代表数量，封装后的TensorDataset可以直接传递给DataLoader
    return data.DataLoader(dataset,batch_size,shuffle = is_train)
    # 将Dataset中的单个样本组合成batchsize的张量，DataLoader必须基于Dataset对象
    # 该函数返回一个可迭代对象（生成器）
def linreg(X,w,b):
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

class Timer:
    # 用来计时
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        # 计时开始
        self.tik = time.time()
        # 返回一个秒数
    def stop(self):
        # 计时结束
        self.times.append(time.time()-self.tik)
        return self.times[-1]
        # 返回最后新添加的值
    def avg(self):
        return sum(self.times())/len(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    def sum(self):
        return sum(self.times)  # 直接对列表求和
    

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    def __init__(self,xlabel = None,ylabel = None, legend = None, xlim = None,
                 ylim = None, xscale = 'linear',yscale = 'linear',
                 fmts = ('-','m--','g-','r:'), nrows = 1, ncols = 1,
                 figsize = (4.5,3.5)):
        if legend is None:
            legend = []
        plt.ion()

        self.fig, self.axes = plt.subplots(nrows,ncols, figsize = figsize)

        if nrows*ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda:set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        # lambd 参数列表：表达式
        # 其函数名和参数列表都可以没有；可以作为参数传递到高等函数里
        
        self.X,self.Y,self.fmts = None, None, fmts
        self.fig.show()
        # 显示窗口
        
    def add(self,x,y):
        if not hasattr(y,'__len__'):
            # hasattr检查一个对象是否具有指定的属性或者方法
            # 检查y是否不能调用len(y)，不能调用就将它放入一个list中
            y = [y]
        n = len(y)
        if not hasattr(x,'__len__'):
            x = [x]* n
        if not self.X:
            self.X = [[] for _ in range(n)]
            # 创建一个n个[]的列表
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i,(a,b) in enumerate(zip(x,y)):
            # enumerate(iterable, start = 0),返回一个enumerate对象，每次迭代生成一个元组（index,value），
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)

        self.config_axes()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
def loadDataFashionMnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    
    if resize:
        trans.insert(0,transforms.Resize(resize))
        # 第一个参数为index，代表新元素的插入位置
        # list(index, elements)，elements代表待插入的元素
    trans = transforms.Compose(trans)
    # transforms.Compose将一系列的transfroms的操作的list表转化为一个可调用的组合变换对象
    # 直接将数据输入，就会依次执行trans里的操作
    mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=False)
    # Fashion-MNIST包含10个类别的图像，训练集6000张图像，测试集1000张
    # 将数据存在根目录下的data文件夹内，如果data不存在，就创建
    # transform表示转化，这里是预处理，根据trans，这里是将数据转化为Tensor
    # download表示如果不存在，则从网络下载
    return (data.DataLoader(mnist_train,batch_size,shuffle=True),data.DataLoader(mnist_test,batch_size,shuffle = False))

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis = 1)
        # argmax(axis = 1)寻找每一行的概率最大值，然后将该数值变成该值的索引，从0开始
    cmp = y_hat.type(y.dtype)==y
    # 将y_hat的数据类型改成真实标签数据y的数据类型，==是逐元素比较的
    return float(cmp.type(y.dtype).sum())
    # 输出数字，有多少的和真实标签一样

def train_epoch_ch3(net,train_iter,loss,updater):
    
    if isinstance(net, torch.nn.Module):
        net.train()
        # 递归地将模型中的所有的子模块的training属性都设置为True，确保整个模型都一致
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            # 在每次参数更新前都要调用，否则梯度累计
            l.mean().backward()
            # 使用mean()可以去掉批大小的影响，因为SGD已经梯度归一化
            updater.step()
            # 在使用backward()计算梯度后，通过step()可以应用梯度下降
        else:
            l.sum().backward()
            updater(X.shape[0])
            
        metric.add(float(l.sum()),accuracy(y_hat, y),y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

def evaluateAccuracy(net,data_iter,w,b):
    if isinstance(net,torch.nn.Module):
        # isinstance(object,class or tuple)用来检测object是否是某个类或者该类的子类的实例
        net.eval()
        # eval()将网络置于评估模式，确保结果一致
        
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X,w,b),y), y.numel())
            # ynumel返回张量中的元素数量
    return metric[0]/metric[1]


def train_ch3(net,train_iter,test_iter,w,b,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim = [0.3,0.9],
                        legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,w,b,loss,updater)
        test_acc = evaluateAccuracy(net,test_iter,w,b)
        animator.add(epoch,train_metrics+(test_acc,))
        print(f'train loss:{train_metrics[0]:.2f} train acc:{train_metrics[1]:.2f}')
        
    train_loss,train_acc = train_metrics
    assert train_loss <0.5,train_loss
    # assert为断言语句，assert condition, message
    # condition为条件，条件不满足的话，会抛出错误，信息为message
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """Plot data points with styles."""
    # plt.show()
    if legend is None:
        legend = []

    # 尝试获取当前的 axes，如果没有则创建一个新的 figure 和 axes
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)

    # Helper function to check if `X` has one axis to plot `Y` vs `X`
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        # 如果 X 是一维的，则 X 作为 x 轴，Y 作为 y 轴
        X = [X] # 统一处理成列表
    if Y is None:
        # 如果只提供了 X，则认为 X 是 y 轴数据，x 轴自动生成 (0, 1, ...)
        X, Y = [[]] * len(X), X # X 变成空列表组成的列表，Y 变成原来的 X
    elif has_one_axis(Y):
        # 如果 Y 是一维的，将其包装成列表，表示只画一条线
        Y = [Y]

    # 如果 X 的数量少于 Y，自动补齐 X
    if len(X) != len(Y):
        X = X * len(Y) # 使用提供的 X 重复绘制多条 Y 线

    # 清除之前的绘图内容（如果 axes 是复用的）
    # axes.cla() # 注释掉这行，以便在同一个 axes 上叠加绘图

    # 循环绘制每一条线
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            # 如果 x 为空，则使用默认的 x 轴（0, 1, 2...）
            axes.plot(y, fmt)

    # 设置坐标轴标签、范围、刻度、图例等
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()
    # 返回 axes 对象，方便进一步操作
    return axes

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def sgd(params, lr, batch_size):
    with torch.no_grad():
        # with是一个上下文管理操作，代表进入某个操作
        # 不管异常与否，with保证在退出的时候执行清理操作
        # torch.no_grad()代表在with代码块中，所有的张量操作的梯度跟踪被关闭，可以减少显存占用
        # 适用于不需要反向传播的过程
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
            # 将参数梯度清零，防止梯度累计


def download_Fra2Eng(name,data_url, cache_dir=os.path.join('..', 'data')):
    DATA_HUB = dict()
    # 创建字典
    DATA_HUB['fra-eng'] = (  # @save
        data_url + 'fra-eng.zip',
    '94646ad1522d915e7b0f9296181140edcf86a4f5')
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    # exist_ok为True代表目录如果存在也不报错，该函数是为了创建一个目录
    fname = os.path.join(cache_dir, url.split('/')[-1])
    # os.path.join(path1, path2, ..., pathN)，将多个path拼接在一起，并且使用正确的分隔符
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        # 创建一个SHA-1哈希算法的对象，初始化了一个SHA-1哈希对象
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            # 缓存命中
            return fname

    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压缩 zip/tar 文件"""
    fname = download_Fra2Eng('fra-eng','http://d2l-data.s3-accelerate.amazonaws.com/')
    # download_Fra2Eng()
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只支持 zip/tar 文件'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def set_figsize(figsize = (3.5,2.5)):
    plt.rcParams['figure.figsize'] = figsize

# PyTorch 版 D2L 的典型实现
# 依赖的辅助函数：read_data_nmt, preprocess_nmt, tokenize_nmt,
#                build_array_nmt, load_array, 以及 d2l.Vocab
def read_data_nmt():
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir,'fra.txt'),'r',encoding='utf-8') as f:
        return f.read()
def preprocess_nmt(text):
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char !=' '
        # set 元素唯一，元素无序，可变，元素必须哈希
        # in函数判断某个元素是否属于某个容器，例如x in y，判断x是否在y内
    text = text.replace('\u202f',' ').replace('\xa0',' ').lower()
    # lower()取小写字母
    # replace('\xa0',' ')用来替换'\xa0'，把不间断空格换成普通空格
    # replace('\u202f', ' ')用来替换'\u202f'，窄不间断空格替换成普通空格
    out = [' ' + char if i > 0 and no_space(char, text[i- 1]) else char
 for i, char in enumerate(text)]
    # 这行代码是为非英文字符前面加一个空格
    return ''.join(out)
    # join函数来将一个可迭代对象，每个元素都是字符串，拼接起来，sep插在元素之间
    # 'sep'.join(iterable)
def tokenize_nmt(text,num_examples=None):
    # 词元化数据集
    source, target = [],[]
    for i,line in enumerate(text.split('\n')):
        if num_examples  and i >num_examples:
            break
        parts = line.split('\t')
        # \t制表符，按了一次tab
        if len(parts) ==2:
            source.append(parts[0].split(' '))
            # str.split(sep = None,maxsplit = -1)，sep为分隔符，maxsplit代表分割次数
            # 返回一个列表list[str]包含分割后的字串，加入不给sep，则默认为任意空白字符

            target.append(parts[1].split(' '))
    return source, target

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回机器翻译数据迭代器和两个词表（源/目标）。"""
    # 读取并预处理原始英法平行语料
    text = preprocess_nmt(read_data_nmt())
    src_tokens, tgt_tokens = tokenize_nmt(text, num_examples)

    # 构建源/目标词表（含保留符号）
    src_vocab = Vocab(src_tokens, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(tgt_tokens, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 张量化并计算有效长度
    src_array, src_valid_len = build_array_nmt(src_tokens, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(tgt_tokens, tgt_vocab, num_steps)

    # 小批量数据迭代器：源句子、源有效长度、目标句子、目标有效长度
    data_iter = load_data((src_array, src_valid_len,
                            tgt_array, tgt_valid_len), batch_size)
    return data_iter, src_vocab, tgt_vocab

def truncate_pad(line, num_steps, padding_tokens):
    # 截断或者填充文本序列
    if len(line)>num_steps:
        return line[:num_steps]
    return line +[padding_tokens] * (num_steps - len(line))


def build_array_nmt(lines,vocab,num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    # 为每个序列末尾添加<eos>，表示序列结束
    array = torch.tensor([truncate_pad(
        l,num_steps,vocab['<pad>']
    ) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 使用TkAgg来弹出一个交互窗口


# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 将当前目录加入 Python 路径




def read_time_machine():
    DATA_HUB = dict()
    DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # [^...]匹配非括号内的字符
    # A-Za-z表示所有的大小写字母
    #


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    # str.split(sep=None, maxsplit=-1)
    # sep（可选）: 分隔符，默认为所有空白字符（如空格、换行、制表符等）。
    # maxsplit（可选）: 最大分割次数，默认为 -1（表示不限制次数）。

    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)

        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # counter.items()输出元组列表，元组为(元素，出现次数)
        # reverse=True表示从高到低，降序排列；key = lambda x:x[1]，按元组的第二个元素排序
        self.idx_to_token = ['<unk>'] + reserved_tokens

        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 将idx_to_token的（token）转化为{token:idx}的字典，其中token为字典的键，idx作为值

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                # 将词添加进token_to_idx列表中，idx为其所在位置

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 判断tokens是否存在，或者是否是list
        tokens = [token for line in tokens for token in line]
        # for line in tokens
        #     for token in line
        # 这个嵌套是这样的，将所有的词展平
    return collections.Counter(tokens)
    # collections.Counter返回一个类似字典的对象，Counter对象
    # {'apple':2,'banana':1}


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    # num_steps是每个子序列中预定义的时间步数（即每一个序列的长度）
    corpus = corpus[random.randint(0, num_steps - 1):]
    # randint给出从0到num_steps-1之间的任意整数，包含0和num_steps-1
    # -1是为了考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 将原始的corpus分成多个子序列
    random.shuffle(initial_indices)

    # 将该序列进行打乱顺序，不一定在原始序列上相邻
    def data(pos):
        # 返回从pos位置开始，长度为num_step的序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # initial_indices为包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 生成顺序子序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    # 加载序列数据的迭代器
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        # 在Python中，__iter__方法是实现迭代器协议的关键方法之一，用于使类的实例成为可迭代对象。以下是详细说明：
        # 定义__iter__方法后，对象可以通过for循环、next()函数、或生成器表达式等进行遍历。
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    H, = state
    outputs = []
    # X的形状为：（批量大小，词表大小）
    for X in inputs:
        H = torch.tanh(torch.mm(X, w_xh) + H @ w_hh)
        Y = H @ w_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):
        # 类obj，创建一下，我们直接运行obj()，会调用obj.__call__
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        # 热编码，将一个有n个数据，h个类别的向量数据，把nx1大小的向量，转换成一个大小为
        # nxh的向量，例如[0,1,2],转换为[[1,0,0],[0,1,0],[0,0,1]]
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 函数名 = lambda 参数:表达式，等价于def函数名（参数）：return 表达式
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
        # argmax和argmin返回最大值或者最小值索引，dim指定维度，keepdim，是否保持维度
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, Timer()
    metric = Accumulator(2)
    for x, y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=x.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
                # detach_()是原地操作，阻止某个张良成为计算图的一部分，避免计算梯度
            else:
                for s in state:
                    s.detach_()
        y = y.T.reshape(-1)
        x, y = x.to(device), y.to(device)
        y_hat, state = net(x, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))
    print("Train completed！")
    plt.ioff()  # 关闭交互模式
    plt.show()


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        # **语法的作用是将后续接收到关键词参数打包成一个字典
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        x = F.one_hot(inputs.T.long(), self.vocab_size)
        x = x.to(torch.float32)
        y, state = self.rnn(x, state)
        output = self.linear(y.reshape((-1, y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))




