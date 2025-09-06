import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import sys
from multi_head_attenuation import MultiHeadAttenuation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

num_hiddens,num_heads = 100,5

attenuation = MultiHeadAttenuation(num_hiddens,num_hiddens,num_hiddens,
                                   num_hiddens,num_heads,0.5)
attenuation.eval()
class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len = 1000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个dropout层
        # 创建一个足够大的矩阵用于存放位置编码
        self.P = torch.zeros((1,max_len,num_hiddens))
        x = torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(
            0,num_hiddens,2,dtype = torch.float32)/num_hiddens)
        self.P[:,:,0::2] = torch.sin(x)
        self.P[:,:,1::2] = torch.cos(x)
    def forward(self,x):
        x = x+self.P[:,:x.shape[1],:].to(x.device)
        return self.dropout(x)
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
plt.figure()
plt.plot(torch.arange(num_steps), P[0, :, 6:10], label="$P_{pos}(pos,0,0)$")
plt.show()
