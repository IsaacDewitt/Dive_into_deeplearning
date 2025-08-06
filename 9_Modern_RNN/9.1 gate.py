import torch
from torch import nn

import utility
import utility_NLP

def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size = shape, device = device) * 0.01
        # 使用一个高斯分布来进行参数的初始化
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device = device))
    W_xz,W_hz,b_z = three()
    # 更新门参数
    W_xr,W_hr,b_r = three()
    # 重置门参数
    W_xh,W_hh,b_h = three()
    # 候选隐状态参数
    W_hq = normal((num_inputs,num_hiddens))
    b_q = torch.zeros(num_outputs,device = device)
    params = [W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
        # inplace操作函数,是否开启梯度运算,即该张量是否参与自动求导
        # 只有浮点类型和复数类型的张量才能设置
    return params

def init_gru_state(batch_size, num_hiddens, devices):
    return (torch.zeros((batch_size,num_hiddens),device = devices),)

def gru(inputs,state,params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for x in inputs:
        Z = torch.sigmoid(x@W_xz +(H@W_hz)+b_z)
        # @在pytorch中是矩阵乘法运算符,等价于torch.matmul
        # *在pytorch中是逐元素相乘(element-wise)
        R = torch.sigmoid(x@W_hr +(H@W_hr)+b_r)
        H_tilda = torch.tanh((x@W_xh)+((R*H)@W_hh)+b_h)
        H = Z* H+(1-Z)*H_tilda
        Y = H@W_hq+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0), (H,)

if __name__ == '__main__':
    batch_size, num_steps = 32,35
    train_iter,vocab = utility_NLP.load_data_time_machine(batch_size,num_steps)
    vocab_size, num_hiddens,device = len(vocab),256,utility.try_gpu()
    # num_epochs,lr = 500,1
    # model = utility_NLP.RNNModelScratch(len(vocab),num_hiddens,device,get_params,
    #                                     init_gru_state, gru)
    # # 需要补充train_ch8
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs,num_hiddens)
    # model =

