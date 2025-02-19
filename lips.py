import torch
from torch import nn
import numpy as np
from torch.nn.functional import gelu
import math

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation = 'ReLU', n_hidden_layers=1,bias=False):
        super(MLP, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'Softplus':
            self.act = nn.Softplus()
        self.l_first = nn.Linear(input_size,hidden_size,bias=bias)
        for i in range(n_hidden_layers-1):
            setattr(self,f'l{i}',nn.Linear(hidden_size,hidden_size,bias=bias))
        self.l_last = nn.Linear(hidden_size,output_size,bias=bias)
    
    def forward(self,x):
        x = self.act(self.l_first(x))
        for i in range(self.n_hidden_layers-1):
            x = getattr(self,f'l{i}')(x)
            x = self.act(x)
        return self.l_last(x)

def M_emb(emb, x = None, p='inf'):
    '''
    compute M(E) = max(max(||E_i||_p, max||x_i - E_j||_p))
    '''
    if p == 'inf':
        p = float(p)
    E = emb.weight
    n = E.shape[0]
    #add zero vector
    if x is not None:
        x = torch.nn.functional.pad(x, (0,1), "constant", 0)
        return torch.cdist(emb(x),E,p=p).max()
    else:
        return torch.cdist(E,E,p=p).max()

def M_conv(conv, p='inf'):
    '''
    compute M(K) = max(max(||K_i||_p, max||K_i - K_j||_p))
    '''
    if p == 'inf':
        p = float(p)
    K = conv.weight.transpose(0,2).transpose(1,2)
    q = K.shape[0]

    M=0
    for i in range(q):
        M += torch.linalg.matrix_norm(K[i,:,:],ord=p)
    return M

def lips_MLP(mlp, p='inf'):
    '''
    compute the local lipschitz constant
    '''
    if p == 'inf':
        p = float(p)

    #print(mlp.l_first.weight)
    L = torch.linalg.matrix_norm(mlp.l_first.weight, ord=p)
    for i in range(mlp.n_hidden_layers-1):
        L *= torch.linalg.matrix_norm(getattr(mlp,f'l{i}').weight, ord=p)
    return L*torch.linalg.matrix_norm(mlp.l_last.weight, ord=p)
    

if __name__ == '__main__':
    n_emb = 80
    l = 200
    d = 512
    k = 100
    q = 5
    x = torch.randn((1,l,d))
    emb = nn.Embedding(n_emb,d)
    conv = nn.Conv1d(d,k,kernel_size=q,stride=1)
    mlp = MLP(d,d,k)
    print(M_emb(emb, p='inf'))
    print(M_conv(conv, p='inf'))
    print(lips_MLP(mlp, p='inf'))