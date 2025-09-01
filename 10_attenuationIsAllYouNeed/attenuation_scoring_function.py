import math
import torch
from torch import nn
from ..utility import sequence_mask

def masked_softmax(x,valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax"""
    # x: 3D张量，valid_lens: 1D或者2D张量
    if valid_lens is None:
        return nn.functional.softmax(x,dim = 1)
        # 沿着列操作，即对每一行向量进行softmax
    else:
        shape = x.shape
        if valid_lens.dim()==1:
            valid_lens = torch.repeat_interleave(valid_lens,shape[1])
            # 如果不指定dim，则先将输入展平为一维，然后进行重复
            # 和numpy.repeat是一回事
        else:
            valid_lens = valid_lens.reshape(-1)
        x = sequence_mask(x.reshape(-1,shape[-1]),valid_lens,value = -1e6)
        return nn.functional.softmax(x.reshape(shape),dim = 1)
    
class AdditiveAttenuation(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttenuation,self).__init__(**kwargs)
        self.w_k = nn.Linear(key_size,num_hiddens,bias = False)
        self.w_q = nn.Linear(query_size,num_hiddens,bias = False)
        self.w_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens):
        queries,keys = self.w_q(queries),self.w_k(keys)
        # queries(batch_size, 查询个数，1，num_hiddens)
        # key的形状(batch_size,1,键值对个数，num_hiddens)
        # 利用广播进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # unsqueeze函数会把原位置及其后的元素都往右挤
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attenuation_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.drop(self.attenuation_weights),values)
    