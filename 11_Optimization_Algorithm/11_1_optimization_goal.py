import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def f(x):
    """风险函数f"""
    return x * torch.cos(np.pi * x)

def g(x):
    """经验风险函数g"""
    # 用来模拟整个训练集的泛化误差
    return f(x) + 0.2 * torch.cos(5* np.pi*x)


x = torch.arange(0.5, 1.5, 0.01)
plt.figure(figsize=(4.5,2.5))
plt.plot(x,f(x),label = 'risk function')
plt.plot(x,g(x),linestyle = '--',label = 'empirical risk')
plt.annotate('min of\nempirical risk', (1.0,-1.2), (0.5,-1.1), arrowprops=dict(arrowstyle='->'))
# matplotlib.pyplot.annotate(text, xy, xytext=None, 
                        #    xycoords='data', textcoords=None,
                        #    arrowprops=None, **kwargs)
# text要注释的文本，xy，二元组，注释的坐标点，xytext，二元组，注释文本的位置，如果不写就在xy点上
plt.annotate('min of risk',(1.1,-1.05),(0.95,-0.5), arrowprops=dict(arrowstyle='->'))
plt.show()