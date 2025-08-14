import torch
from torch import nn

import utility
import utility_NLP
from utility_NLP import RNNModel


def get_lstm_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))
    w_xi, w_hi,b_i = three()
    w_xf,w_hf,b_f = three()
    w_xo,w_ho,b_o = three()
    w_xc,w_hc,b_c = three()

    w_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    params = [w_xi,w_hi,b_i,w_xf,w_hf,b_f,w_xo,w_ho,b_o,w_xc,w_hc,b_c,w_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device = device),
            torch.zeros((batch_size,num_hiddens),device = device))

def lstm(inputs,state,params):
    [w_xi,w_hi,b_i,w_xf,w_hf,b_f,w_xo,w_ho,b_o,w_xc,w_hc,b_c,w_hq,b_q] = params
    (H,C) = state
    outputs = []
    for x in inputs:
        I = torch.sigmoid((x@w_xi) + (H@w_hi) + b_i)
        F = torch.sigmoid((x@w_xf) + (H@w_hf) + b_f)
        O = torch.sigmoid((x@w_xo) + (H@w_ho) + b_o)
        C_tilda = torch.tanh((x@w_xc) + (H@w_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H@w_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim = 0), (H,C)

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = utility_NLP.load_data_time_machine(batch_size,num_steps)
    vocab_size,num_hiddens,device = len(vocab), 256, utility.try_gpu()
    num_epochs,lr = 500,1
    # model = utility_NLP.RNNModelScratch(len(vocab),num_hiddens,device,get_lstm_params,
    #                                     init_lstm_state,lstm)
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs,num_hiddens)
    model = RNNModel(lstm_layer,len(vocab))
    model = model.to(device)
    utility_NLP.train_ch8(model,train_iter,vocab,lr,num_epochs,device)

