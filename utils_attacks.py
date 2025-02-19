import numpy as np
import torch
import string
import torch.nn.functional as F

from utils import collate_fn, generate_all_sentences


'''
Charmer related
------------------------------------------------------------------------------------------------------------------
'''

def margin_loss_lm(logits, true_class):
    '''
    Standard margin loss for classification
    '''
    #maximum different than true class
    max_other,_ = (torch.cat((logits[:,:true_class], logits[:,true_class+1:]), dim=-1)).max(dim=-1)
    return max_other - logits[:,true_class]

class margin_loss_lm_batched():
    def __init__(self,reduction = 'None'):
        self.reduction = reduction
    
    def __call__(self,logits, true_classes):
        '''
        Standard margin loss for classification
        '''
        L = torch.cat([margin_loss_lm(l.unsqueeze(0), t) for l,t in zip(logits,true_classes)], dim=0)
        if self.reduction == 'mean':
            return torch.mean(L)
        elif self.reduction == 'sum':
            return torch.sum(L)
        else:
            return L

def attack_text_charmer_classification(model,tokenizer,to_id,pad,sentence,label,device,n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False):
    '''
    n in this case is the number of positions in charmer
    '''
    criterion = margin_loss_lm_batched(reduction='None')

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #Select best positions
            VV = [ord(' ')]
            SS = generate_all_sentences(sentence,VV,alternative=-1)
            X = [{'sentence' : s, 'label' : label} for s in SS]
            data = torch.utils.data.DataLoader(X, batch_size = 512, shuffle = False, collate_fn = lambda x: collate_fn(x,to_id,pad=pad))
            outs = []
            for x in data:
                s = x[0].to(device)
                out,_ = model(s)
                outs.append(out)
            out = torch.cat(outs,dim=0)

            # #Cut to length pad, tokenize and pad output tokens            
            loss = criterion(out,label*torch.ones(len(SS),device=device).long().to(device))

            top_positions = torch.topk(loss,min(n,out.shape[0]),dim=0).indices
            
            #Generate all possible sentences with the top positions
            SS = generate_all_sentences(sentence,V,subset_z=top_positions,alternative=-1)
            X = [{'sentence' : s, 'label' : label} for s in SS]
            data = torch.utils.data.DataLoader(X, batch_size = 512, shuffle = False, collate_fn = lambda x: collate_fn(x,to_id,pad=pad))
            outs = []
            for x in data:
                s = x[0].to(device)
                #l = x[1].to(device)
                out,_ = model(s)
                outs.append(out)
            out = torch.cat(outs,dim=0)
            loss = criterion(out,label*torch.ones(len(SS),device=device).long().to(device))

            sentence = SS[torch.argmax(loss).item()]

            text_probs = (out[torch.argmax(loss).item(),:]).softmax(dim=-1)
            if text_probs.argmax().item() != label:
                break
        return sentence,dist+1

'''
randomized
'''

def sample_deletes(S,n=1,p_del = 0.5):
    '''
    Randomly deletes characters from a string
    '''
    return [''.join([c for c in S if np.random.rand() > p_del]) for _ in range(n)]


if __name__ == '__main__':
    #Test generate_sentence
    S = 'hello'
    print(S,sample_deletes(S,n=10,p_del = 0.5))

    x = torch.randn(10,2)
    print(torch.mean(F.one_hot(x.argmax(dim=-1),num_classes=x.shape[-1]).float(),dim=0))