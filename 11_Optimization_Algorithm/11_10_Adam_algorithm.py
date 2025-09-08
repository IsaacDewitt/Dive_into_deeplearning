import torch

def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim,1)),torch.zeros(1)
    
    s_w, s_b = torch.zeros((feature_dim,1)),torch.zeros(1)
    return ((v_w,s_w), (v_b,s_b))

def adam(params, states,hyparams):
    beta1,beta2, eps = 0.9,0.999,1e-6
    for p,(v,s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            # vt <- beta1 * vt-1 + (1-beta1) * gt
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            #  st <- beta2 * st-1 + (1-beta2) * gs
            v_bias_corr = v/ (1-beta1**hyparams['t'])
            # 两个偏置项
            s_bias_corr = s/ (1-beta2 ** hyparams['t'])
            p[:]-=hyparams['lr'] * v_bias_corr/(torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyparams['t'] +=1