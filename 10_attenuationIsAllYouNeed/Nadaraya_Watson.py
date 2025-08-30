import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utility import show_heatmaps,Animator

class NWKernelRegression(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,),requires_grad=True))

    def forward(self,queries, keys,values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries -keys) * self.w)**2/2,dim=1
        )

        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
        # bmm,batch matrix-matrix multiplication批量矩阵乘法
        # 对于一批矩阵进行矩阵乘法运算，torch.bmm(input, mat2, *, out=None) -> Tensor
        # input为一个三维张量，形状为(b,n,m)大小为b，每个矩阵大小为nxm
        # mat2，大小为b,m,p, mxp

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def Gaussian_kernel_aattenuation(xi,x,y):
    return np.sum(softmax(-0.5*(xi-x)**2)*y)
    
def Gaussian_kernel_aattenuation_matrix(x,xtrain,ytrain):
    return softmax(-0.5*(x-xtrain)**2)*ytrain

def GKAM_B(x,xtrain,ytrain):
    temp = -0.5*(x-xtrain)**2
    temp = np.exp(temp)
    divf = np.sum(temp,axis=1).reshape(100,1)
    # 这里最傻逼的是axis = 0代表是列求和，axis = 1代表是行求和
    return (temp/divf)*ytrain
    

if __name__ == "__main__":
    num_point = 100
    x_train = np.sort(np.random.random(num_point)*5)
    kesai = np.random.normal(loc = 0, scale=0.5, size = num_point)
    y_train = 2*np.sin(x_train)+x_train**0.8+kesai
    x_test = np.arange(0,5,0.05)
    y_truth = 2*np.sin(x_test) + x_test**0.8
    x_test_repeat = np.repeat(x_test[:,np.newaxis],num_point,1)
    # numpy里的广播机制，numpy的广播机制是两个向量维度不同的时候，小的那个形状会在最左边填加一个为维度
    # 即(100,)变成(1,100)，然后沿着维度为1的地方进行重复，直到满足计算
    y_hat = [Gaussian_kernel_aattenuation(item,x_train,y_train) for item in x_test]
    attention_matrix = np.ones((num_point,num_point))
    for idx,item in enumerate(x_test):
        attention_matrix[idx,:] = Gaussian_kernel_aattenuation_matrix(item,x_train,y_train)
    am = torch.tensor(GKAM_B(x_test_repeat,x_train,y_train))
    # 这里x_train为(100,)，y_train大小为(100,)，
    attention_matrix = torch.tensor(attention_matrix).unsqueeze(0).unsqueeze(0)
    am = am.unsqueeze(0).unsqueeze(0)
    # plt.figure()
    # plt.scatter(x_train,y_train,marker = 'o',alpha = 0.5,label = "train")
    # plt.plot(x_train,y_truth,color = 'b',label = "truth")
    # plt.plot(x_test,y_hat,linestyle = '--',label = "pred")
    # plt.legend()
    
    # show_heatmaps(attention_matrix,xlabel='Sorted training inputs',ylabel='Sorted testing inputs')
    # show_heatmaps(am,xlabel='Sorted training inputs',ylabel='Sorted testing inputs')
    # show_heatmaps(am-attention_matrix,xlabel='Sorted training inputs',ylabel='Sorted testing inputs')
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test)
    xtr = x_train.repeat((num_point,1))
    ytr = y_train.repeat((num_point,1))
    keys = xtr[(1-torch.eye(num_point)).type(torch.bool)].reshape((num_point,-1))
    # 不选对角位置，每一行都缺一个，即第[n,n]个元素，所以大小为(num_point,num_point-1)
    values = ytr[(1-torch.eye(num_point)).type(torch.bool)].reshape((num_point,-1))
    # print(xtr[:,1])
    # print(x_test_repeat[1,:])
    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(),lr = 0.5)
    for epoch in tqdm.tqdm(range(5)):
        trainer.zero_grad()
        l = loss(net(x_train,keys,values),y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch+1}, loss {float(l.sum()): .6f}')
    keys = x_train.repeat((num_point,1))
    values = y_train.repeat((num_point,1))
    y_hat = net(x_test,keys,values).unsqueeze(1).detach()
    plt.figure()
    plt.scatter(x_train,y_train,marker = 'o',alpha = 0.5,label = "train")
    plt.plot(x_train,y_truth,color = 'b',label = "truth")
    plt.plot(x_test,y_hat,linestyle = '--',label = "pred")
    plt.legend()
    show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                 ylabel='Sorted testing inputs')
    print(net.w.shape)
    plt.show()
