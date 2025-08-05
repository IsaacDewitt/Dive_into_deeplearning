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
from IPython import display
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import os
import requests

# from matplotlib import rcParams

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DATA_HUB = dict()
# 创建字典
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = ( #@save
 DATA_URL + 'kaggle_house_pred_train.csv',
 '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = ( #@save
 DATA_URL + 'kaggle_house_pred_test.csv',
 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def download(name,cache_dir = os.path.join('..','data')):
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
    
def use_svg_display():
    """使用svg格式在Jupyter中显示绘图

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

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
                 fmts = ('-','m--','g-','r:'),nrows = 1,ncols = 1,
                 figsize = (3.5,2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows,ncols, figsize = figsize)
        if nrows*ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda:set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        # lambd 参数列表：表达式
        # 其函数名和参数列表都可以没有；可以作为参数传递到高等函数里
        
        self.X,self.Y,self.fmts = None, None, fmts
        
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
        display.display(self.fig)
        display.clear_output(wait = True)
        
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

def set_figsize(figsize=(3.5, 2.5)):  # :contentReference[oaicite:3]{index=3}
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """Plot data points with styles."""
    # 输入数据处理
    # if Y is None:
    #     X, Y = [[]] * len(X), X
    # elif not isinstance(X[0], (list, tuple)):
    #     X = [X]
    # if not isinstance(Y[0], (list, tuple)):
    #     Y = [Y]
    # if len(X) != len(Y):
    #     X = X * len(Y)
        
    # # 设置绘图样式
    # rcParams['font.size'] = 14  # 全局字体大小
    # plt.rcParams['figure.figsize'] = figsize  # 默认图像尺寸
    
    # # 创建坐标轴
    # if axes is None:
    #     axes = plt.gca()
    # axes.cla()
    
    # # 绘制每条曲线
    # for x, y, fmt in zip(X, Y, fmts):
    #     if x:
    #         axes.plot(x, y, fmt)
    #     else:
    #         axes.plot(y, fmt)
    
    # # 设置坐标轴属性
    # axes.set_xlabel(xlabel)
    # axes.set_ylabel(ylabel)
    # axes.set_xscale(xscale)
    # axes.set_yscale(yscale)
    # axes.set_xlim(xlim)
    # axes.set_ylim(ylim)
    
    # # 添加图例
    # if legend:
    #     axes.legend(legend)
    # axes.grid()  # 显示网格

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

# --- 示例用法 ---
if __name__ == '__main__':
    # 示例 1: 绘制 y = x^2
    x = np.arange(0, 3, 0.1)
    plot(x, [x**2, 2*x - 1], 'x', 'f(x)', legend=['f(x) = x^2', 'f(x) = 2x - 1'])
    plt.show()

    # 示例 2: 只提供 Y 值
    y_data = np.random.rand(10)
    plot(y_data, ylabel='Random Values')
    plt.show()

    # 示例 3: 在同一个 axes 上绘制
    fig, axes = plt.subplots(figsize=(6, 4))
    plot(x, x**2, xlabel='x', ylabel='y', axes=axes, legend=['x^2'])
    plot(x, np.sin(x) * 5, axes=axes, legend=['5sin(x)'], fmts=('g--')) # 使用不同的格式符
    plt.show()
    

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











