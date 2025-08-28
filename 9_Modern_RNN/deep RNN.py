import torch
from torch import nn

from utility import try_gpu,load_data_time_machine, RNNModel, train_ch8

if __name__ == '__main__':
    batch_size = 32
    num_steps = 35
    train_iter,vocab = load_data_time_machine(batch_size,num_steps)
    vocab_size, num_hiddens, num_layer = len(vocab),256,2
    num_inputs = vocab_size
    device = try_gpu()
    lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layer)
    model = RNNModel(lstm_layer,len(vocab))
    model = model.to(device)
    num_epochs,lr = 500,2
    train_ch8(model,train_iter,vocab,lr*1.0, num_epochs,device)

