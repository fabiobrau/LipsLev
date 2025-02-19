import torch
from torch import nn
import numpy as np
from torch.nn.functional import one_hot
import lips
from utils import get_all_replacements

class HHAct(nn.Module):
    def __init__(self, input_size):
        super(HHAct, self).__init__()
        self.w = nn.Linear(input_size,1, bias=False)
    
    def forward(self,x):
        norm_w = torch.norm(self.w.weight,p=2)

        dotp = (self.w(x.transpose(1,2))/norm_w).transpose(1,2)

        return x*(dotp > 0) + (x - 2*(self.w.weight.unsqueeze(-1)/norm_w)*dotp)*(dotp <= 0)

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

class ConvModel(nn.Module):
    def __init__(self, n_chars, embed_size=256, hidden_size=256, n_classes=10, kernel_size = 5, n_layers = 2,p=2, reduce = 'mean', lips_emb = True):
        super(ConvModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_chars = n_chars
        self.kernel_size = kernel_size
        self.emb = nn.Embedding(n_chars, embed_size,padding_idx=0)
        self.n_layers = n_layers
        self.reduce = reduce
        self.lips_emb = lips_emb
        for i in range(self.n_layers):
            if i==0:
                setattr(self,f'conv{i}',nn.Conv1d(embed_size,hidden_size,kernel_size=self.kernel_size, padding=self.kernel_size-1,stride=1, bias=False))
            else:
                setattr(self,f'conv{i}',nn.Conv1d(hidden_size,hidden_size,kernel_size=self.kernel_size, padding=self.kernel_size-1,stride=1, bias=False))
        self.act = nn.ReLU()
        self.C = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.p=p
        if self.p == 'inf':
            self.p = float(self.p)
        self.init_1lips()

    def init_1lips(self):
        with torch.no_grad():
            self.emb.weight/=lips.M_emb(self.emb,p=self.p)

            for i in range(self.n_layers):
                getattr(self,f'conv{i}').weight/=lips.M_conv(getattr(self,f'conv{i}'),p=self.p)

            if self.p==1:
                q = float('inf')
            elif self.p==2:
                q = 2
            elif self.p==float('inf'):
                q = 1

            self.C.weight/=lips.M_emb(self.C,p=q)
        
        
    def compute_lips(self,x, target=0, label=1):
        '''
        compute the local lipschitz constant evaluated at x
        '''
        with torch.no_grad():
            if self.lips_emb:
                ME = lips.M_emb(self.emb,x=x,p=self.p)
            else:
                ME = 1
            MK = 1
            for i in range(self.n_layers):
                MK *=lips.M_conv(getattr(self,f'conv{i}'),p=self.p)
            
        

            if self.p==1:
                q = float('inf')
            elif self.p==2:
                q = 2
            elif self.p==float('inf'):
                q = 1
            if self.reduce == 'sum':
                return (torch.norm(self.C.weight[target] - self.C.weight[label],p=q)*MK*ME)
            elif self.reduce == 'mean':
                return (torch.norm(self.C.weight[target] - self.C.weight[label],p=q)*MK*ME)*2/x.shape[1]
    
    def compute_lips_reg(self,x, target=0, label=1):
        '''
        compute the local lipschitz constant evaluated at x
        '''
        if self.lips_emb:
            ME = lips.M_emb(self.emb,p=self.p)
        else:
            ME = 1
        
        MK = 1
        for i in range(self.n_layers):
            MK *=lips.M_conv(getattr(self,f'conv{i}'),p=self.p)

        if self.p==1:
            q = float('inf')
        elif self.p==2:
            q = 2
        elif self.p==float('inf'):
            q = 1

        return (lips.M_emb(self.C,p=q)*MK*ME)

    def forward(self, x, mask=None):
        bs, length = x.shape[0], x.shape[1]
        if mask is None:
            mask = torch.ones([bs,length,1],device=x.device)
        x = (mask*self.emb(x)).transpose(1,2)
        for i in range(self.n_layers):
            x = self.act(getattr(self,f'conv{i}')(x))
        x = x.transpose(1,2)

        if self.reduce == 'sum':
            x = x.sum(dim=1)
        elif self.reduce == 'mean':
            x = x.mean(dim=1)
        out = self.C(x)
        return out, mask.unsqueeze(-1)
    
    def forward_bounds_till_reduce(self, x, X, delta):
        '''
        Used for computing the IBP bounds of (Huang et al., 2019)
        '''
        z = (self.emb(x) + delta*(self.emb(X)-self.emb(x))).transpose(1,2)
        for i in range(self.n_layers):
            z = self.act(getattr(self,f'conv{i}')(z))
        z = z.transpose(1,2)

        if self.reduce == 'sum':
            z = z.sum(dim=1)
        elif self.reduce == 'mean':
            z = z.mean(dim=1)

        return z.min(axis=0,keepdim=True)[0],z.max(axis=0,keepdim=True)[0]
    
    def bounds_logits(self,l,u,y):
        '''
        Used for computing the IBP bounds of (Huang et al., 2019)
        '''
        Cp = ((self.C.weight > 0)*self.C.weight).transpose(0,1)
        Cn = ((self.C.weight < 0)*self.C.weight).transpose(0,1)

        L = l@Cp + u@Cn
        U = u@Cp + l@Cn
        y_one_hot = one_hot(y,num_classes=self.n_classes)
        return L*y_one_hot + U*(1-y_one_hot)

class ConvLipsModel(nn.Module):
    '''
    Same as ConvModel but enforced to be 1-Lipschitz
    '''
    def __init__(self, n_chars, embed_size=256, hidden_size=256, n_classes=10, kernel_size = 5, n_layers = 2,p=2, reduce = 'mean', lips_emb = True, batch_norm = False, activation = 'ReLU'):
        super(ConvLipsModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_chars = n_chars
        self.kernel_size = kernel_size
        self.emb = nn.Embedding(n_chars, embed_size,padding_idx=0)
        self.n_layers = n_layers
        self.reduce = reduce
        self.lips_emb = lips_emb
        self.batch_norm = batch_norm
        for i in range(self.n_layers):
            if self.batch_norm:
                setattr(self,f'bn{i}', nn.BatchNorm1d(hidden_size, affine=False))
            if i==0:
                setattr(self,f'conv{i}',nn.Conv1d(embed_size,hidden_size,kernel_size=self.kernel_size,stride=1, padding=self.kernel_size-1, bias=False))
            else:
                setattr(self,f'conv{i}',nn.Conv1d(hidden_size,hidden_size,kernel_size=self.kernel_size,stride=1, padding=self.kernel_size-1, bias=False))
        
        if activation == 'ReLU':
            self.act = nn.ReLU()
            self.lips_act = 1
        elif activation == 'softplus':
            self.act = nn.Softplus()
            self.lips_act = 1
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.lips_act = 1/4
        elif activation == 'tanh':
            self.act = nn.Tanh()
            self.lips_act = 1
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=1)
            self.lips_act = 1
        elif activation == 'GELU':
            self.act = nn.GELU()
            self.lips_act = 1
        elif activation == 'HH':
            self.act = HHAct(self.hidden_size)
            self.lips_act = 1
        self.C = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.p=p
        self.p = float(self.p)

    def compute_lips(self,x, target=0, label=1):
        '''
        compute the local lipschitz constant evaluated at x
        '''
        with torch.no_grad():
            if self.p==1:
                q = float('inf')
            elif self.p==2:
                q = 2
            elif self.p==float('inf'):
                q = 1
            L = 1
            if self.reduce == 'sum':
                if self.lips_emb:
                    return L*(lips.M_emb(self.emb,x=x,p=self.p)/lips.M_emb(self.emb,p=self.p))*torch.norm(self.C.weight[target] - self.C.weight[label],p=q)/lips.M_emb(self.C,p=q)

                else:
                    return L*torch.norm(self.C.weight[target] - self.C.weight[label],p=q)/lips.M_emb(self.C,p=q)

            elif self.reduce == 'mean':
                return L*2/x.shape[1]*(lips.M_emb(self.emb,x=x,p=self.p)/lips.M_emb(self.emb,p=self.p))*torch.norm(self.C.weight[target] - self.C.weight[label],p=q)/lips.M_emb(self.C,p=q)
    
    def compute_lips_reg(self,x, target=0, label=1):
        '''
        compute the local lipschitz constant evaluated at x
        '''
        return 1

    def forward(self, x, mask=None):
        bs, length = x.shape[0], x.shape[1]
        if mask is None:
            mask = torch.ones([bs,length,1],device=x.device)
        if self.lips_emb:
            ME = lips.M_emb(self.emb,p=self.p)
        else:
            ME = 1
        x = (mask*self.emb(x)).transpose(1,2)/ME
        for i in range(self.n_layers):
            MK =lips.M_conv(getattr(self,f'conv{i}'),p=self.p)
            x = getattr(self,f'conv{i}')(x/(MK))
            if self.batch_norm:
                MB = torch.max(1/torch.sqrt(getattr(self,f'bn{i}').running_var + getattr(self,f'bn{i}').eps))
                x = getattr(self,f'bn{i}')(x/MB)
            x = self.act(x)/self.lips_act
        x = x.transpose(1,2)

        if self.reduce == 'sum':
            x = x.sum(dim=1)
        elif self.reduce == 'mean':
            x = x.mean(dim=1)
        if self.p==1:
            q = float('inf')
        elif self.p==2:
            q = 2
        elif self.p==float('inf'):
            q = 1
        MC = lips.M_emb(self.C,p=q)
        out = self.C(x/MC)
        return out, mask.unsqueeze(-1)
    
    def forward_bounds_till_reduce(self, x, X, delta):
        '''
        Used for computing the IBP bounds of (Huang et al., 2019)
        '''
        if self.lips_emb:
            ME = lips.M_emb(self.emb,p=self.p)
        else:
            ME = torch.max(torch.linalg.norm(self.emb.weight,dim=1,p=self.p))
        z = (self.emb(x) + delta*(self.emb(X)-self.emb(x))).transpose(1,2)/ME
        for i in range(self.n_layers):
            MK =lips.M_conv(getattr(self,f'conv{i}'),p=self.p)
            z = self.act(getattr(self,f'conv{i}')(z/MK))/self.lips_act
        z = z.transpose(1,2)

        if self.reduce == 'sum':
            z = z.sum(dim=1)
        elif self.reduce == 'mean':
            z = z.mean(dim=1)

        return z.min(axis=0,keepdim=True)[0],z.max(axis=0,keepdim=True)[0]
    
    def bounds_logits(self,l,u,y):
        '''
        Used for computing the IBP bounds of (Huang et al., 2019)
        '''
        Cp = ((self.C.weight > 0)*self.C.weight).transpose(0,1)
        Cn = ((self.C.weight < 0)*self.C.weight).transpose(0,1)

        if self.p==1:
            q = float('inf')
        elif self.p==2:
            q = 2
        elif self.p==float('inf'):
            q = 1
        MC = lips.M_emb(self.C,p=q)
        L = (l@Cp + u@Cn)/MC
        U = (u@Cp + l@Cn)/MC
        y_one_hot = one_hot(y,num_classes=self.n_classes)
        return L*y_one_hot + U*(1-y_one_hot)
    
if __name__ == '__main__':
    model = ConvModel(4, embed_size=150, hidden_size=10, n_classes=2, kernel_size = 5, n_layers = 1,p='inf', reduce = 'mean')
    sentence = 'happy'
    char_to_id = {'h':0, 'a':1, 'p': 2, 'y':3}
    S = get_all_replacements(sentence,char_to_id)
    y = torch.tensor([1])
    print(S)
    x = torch.tensor([char_to_id[c] for c in sentence]).unsqueeze(0)
    X = torch.tensor([[char_to_id[c] for c in s] for s in S])
    print(x.shape,X.shape)
    l,u = model.forward_bounds_till_reduce(x,X,delta=1)
    print(l,u)

    y_ = model.bounds_logits(l,u,y)
    print(y_)