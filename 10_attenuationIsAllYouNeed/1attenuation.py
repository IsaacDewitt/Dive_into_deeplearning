import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utility
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='Reds'):
    utility.use_svg_display()
    num_rows, num_cols = matrices.shape[0],matrices.shape[1]
    fig,axes = plt.subplots(num_rows,num_cols,figsize = figsize,sharex = True,
                            sharey = True, squeeze = False)
   # fig是figure对象，axes，Axes对象
   # axes是一个matplotlib.axes._axes.Axes对象,类似数组,如果是一个单独的子图,返回单个对象
   # 多个子图,返回一个numpy数组,其中包含多个Axes对象,代表的是一个坐标系,例如fig, axes = plt.subplots(2, 2)
   # axes是一个2x2的Numpy数组

    for i,(row_axes, row_matrices) in enumerate(zip(axes,matrices)):
        for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
            # zip把多个可迭代对象打包在一起，返回一个zip对象，第i个是所有组合对象的第i个元素的元组
            pcm = ax.imshow(matrix.detach().numpy(), cmap = cmap)
            if i == num_rows -1:
               ax.set_xlabel(xlabel)
            if j ==0:
               ax.set_ylabel(ylabel)
            if titles:
               ax.set_title(titles[j])
            fig.colorbar(pcm,ax = axes,shrink = 0.6)
    plt.show()

def plot_kernel_reg(x_test,y_hat,x_train,y_train,y_truth):
    utility.plot(x_test,[y_truth,y_hat],'x','y',legend = ['Truth','Pred'],
                 xlim=[0,5],ylim=[-1,5])
    plt.plot(x_train, y_train,'o',alpha = 0.5)
    plt.show()

attenuation_weights = torch.eye(10).reshape((1,1,10,10))
# show_heatmaps(attenuation_weights,xlabel = 'Keys', ylabel = 'Querries')

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
def f(x):
    return 2 * torch.sin(x) + x**0.8
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0,5,0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(n_test)
y_hat = torch.repeat_interleave(y_train.mean(),n_test)
# repeat_interleave沿指定维度重复张量中的元素，返回一个新张量
# torch.repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor
# repeats (int 或 Tensor)如果是 int：表示每个元素都重复 repeats 次。
# 如果是 1-D Tensor：必须和 input 在 dim 维的长度相同，表示该维度的每个元素分别重复多少次。
# dim是维度,返回一个tensor,其形状和input相同,沿着dim重复了指定的次数
plot_kernel_reg(x_test,y_hat,x_train,y_train,y_truth)