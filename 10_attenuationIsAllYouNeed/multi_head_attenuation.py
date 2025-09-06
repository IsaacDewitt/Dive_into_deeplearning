import math
import torch
from torch import nn

class DotProductAttenuation(nn.Module):
    """点积注意力"""
    def __init__(self,dropout,
                 **kwargs):
        super(DotProductAttenuation,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens = None):
        # queries: (batch_size, num_queries, num_hiddens)
        d = queries.shape[-1]
        # batch_size
        scores = torch.bmm(queries,keys.transpose(1,2))//math.sqrt(d)
        # transpose交换两个维度,这里交换第二和第三个
        # bmm实现批量矩阵乘法，[batch_size,num_queries,num_hiddens] * [batch_size,num_hiddens,num_keys]
        # return [batch_size,num_queries,num_keys]
        self.attention_weights = nn.functional.softmax(scores,valid_lens)
        # 这里valid_lens用来指定dim
        return torch.bmm(self.dropout(self.attention_weights),values)
    

class MultiHeadAttenuation(nn.Module):
    """多头注意力的实现"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias = False,
                 **kwargs):
        super(MultiHeadAttenuation,self).__init__(**kwargs)
        self.num_heads = num_heads
        # 头数
        self.attenutaion = DotProductAttenuation(dropout)
        # 用点积注意力计算注意力
        self.W_q = nn.Linear(query_size,num_hiddens,bias = bias)
        # 用于查询，输入(query_size,),输出为(num_hiddens,)
        self.W_k = nn.Linear(key_size,num_hiddens,bias = bias)
        # 用于键，输入(key_size,),输出为(num_hiddens,)
        self.W_v = nn.Linear(value_size,num_hiddens,bias = bias)
        # 用于值，输入(value_size,),输出为(num_hiddens,)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias = bias)
        # 用于输出，输入(num_hiddens,),输出为(num_hiddens,)

    def forward(self,queries,keys,values,valid_lens):
        # queries，keys和values的大小为（batch,查询或者键值对的个数，num_hiddens）
        # valid_lens的形状：（batch,查询或者键值对的个数）
        # 输出的形状为（batch*num_heads,查询或者键值对的个数，num_hiddens/num_heads）
        queries = transpose_qkv(self.W_q(queries),self.num_heads)
        queries = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)
        if valid_lens is not None:
            # 在轴0，将第一项复制num_heads次
            # 然后复制第二项，诸如此类
            valid_lens = torch.repeat_interleave(valid_lens,self.num_heads,dim = 0)
            # 这里valid_lens的形状为(batch*num_heads,查询或者键值对的个数,num_hiddens/num_heads)
        output = self.attenutaion(queries,keys,values,valid_lens)
        output_concat = transpose_output(output,self.num_heads)
        return self.W_o(output_concat)
    
def transpose_qkv(x,num_heads):
    # 输入的x的形状为（batch_szie,查询或者键值对的个数，num_hiddens）
    # 输出的形状为（batch_size,查询或者键值对的个数，num_heads,num_hiddens/num_heads)
    x = x.reshape(x.shape[0],x.shape[1],num_heads,-1)
    x = x.permute(0,2,1,3)
    return x.reshape(-1,x.shape[-1],x.shape[3])

def transpose_output(x,num_heads):
    # 输入的x的形状为（batch_size,查询或者键值对的个数，num_heads,num_hiddens/num_heads)
    # 逆转transpose_qkv函数的操作
    x = x.reshape(-1,num_heads,x.shape[1],x.shape[2])
    x = x.permute(0,2,1,3)
    return x.reshape(x.shape[0],x.shape[1],-1)
if __name__ == '__main__':
    # 测试代码
    num_hiddens, num_heads = 100, 5
    attenuation = MultiHeadAttenuation(num_heads,num_hiddens,num_hiddens,
                                    num_hiddens,0.5)
    attenuation.eval()
    batch_size, num_queries = 2,4
    num_kvpairs, valid_lens = 6, torch.tensor([3,2])
    x = torch.ones((batch_size,num_queries,num_hiddens))
    y = torch.ones((batch_size,num_kvpairs,num_hiddens))
    print(attenuation(x,y,y,valid_lens).shape)