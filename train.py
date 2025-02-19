import os
import json
import torch
import argparse
import pandas as pd
from torch import optim
from utils import train, test, get_dataloaders, char_tokenizer, word_tokenizer, seed_everything
from models import ConvModel, ConvLipsModel
from verification import verify, verified_acc_bruteforce, verified_acc_ibp_replace, verified_acc_ibp

torch.autograd.set_detect_anomaly(True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of a shallow transformer')
    parser.add_argument('--dataset', default='sst2', type=str, help='glue datase or numbers_letters')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--output_folder', default='results', type=str, help='default output folder')
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--hidden_size', default=100, type=int, help='hidden size')
    parser.add_argument('--embed_size', default=150, type=int, help='embedding size')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size for conv model')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers for the convolutional model')
    parser.add_argument('--lr', default=1, type=float, help='Learning rate')
    parser.add_argument('--grad_clip', default='inf', type=float, help='gradient cliping parameter')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lm_weight', default=0, type=float, help='weight for the language modelling loss')
    parser.add_argument('--masking_prob', default=0, type=float, help='probability of masking a char')
    parser.add_argument('--lips_reg', default=0, type=float, help='regularization term for the lipschitz constant')
    parser.add_argument('--model_name', default='ConvLips', type=str, help='name of the architecture to train')
    parser.add_argument('--p', default='inf', help='p norm for the erp distance')
    parser.add_argument('--optimizer', default='SGD', type=str, help='name of the optimizer')
    parser.add_argument('--scheduler', default='cyclic', type=str, help='name of the scheduler')
    parser.add_argument('--steps_decrease', default=1000000, type=int, help='steps to decrease the learning_rate in the scheduler')
    parser.add_argument('--reduce', default='sum', type=str, help='reduction layer (sum, mean, max...)')
    parser.add_argument('--eval_oracle', default=False, type=bool, help='evaluate verified accuracy by bruteforce')
    parser.add_argument('--eval_ibp', default=False, type=bool, help='evaluate verified accuracy by ibp')
    parser.add_argument('--valid_size', default=1000, type=int, help='samples for the validation set')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--word_level', default=False, type=bool, help='train the model at word-level')
    parser.add_argument('--batch_norm', default=False, type=bool, help='use batch normalization')
    parser.add_argument('--synonyms_path', default='synonyms.json', type=str, help='path of the json file with synonyms')
    parser.add_argument('--activation', default='ReLU', type=str, help='activation name')

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.p != 'inf':
        args.p = float(args.p)

    tokenizer = None
    if args.dataset in  ['ag_news', 'imdb', 'fake-news']:
        pad = 1000
    else:
        pad = 286
    synonyms = None
    if args.word_level:
        with open(args.synonyms_path) as f:
            synonyms = json.load(f)

    if args.word_level:
        df_count_emb = pd.read_csv('counter-fitted-vectors.txt', sep=' ', engine='python')
        vocab = df_count_emb.iloc[:,0].values
        # +1 for the unk token
        n_char = len(vocab)+1
        args.embed_size = df_count_emb.shape[1]-1
        embeddings = torch.tensor(df_count_emb.iloc[:,1:].values)
        embeddings = torch.cat([torch.zeros(1, args.embed_size), embeddings], dim=0)
        word_to_id = {word: i+1 for i, word in enumerate(vocab)}
        _, train_loader, valid_loader, test_loader = get_dataloaders(args.dataset,args.batch_size, word_level=True, word_to_id= word_to_id,pad=pad, valid_size=args.valid_size)

    else:
        char_to_id, train_loader, valid_loader, test_loader = get_dataloaders(args.dataset,args.batch_size,pad=pad, valid_size=args.valid_size)
        n_char = len(char_to_id.keys())
    
    if args.model_name == 'Conv':
        net = ConvModel(n_char+1, args.embed_size, args.hidden_size, args.n_classes, kernel_size = args.kernel_size, n_layers = args.n_layers,p=args.p, reduce=args.reduce, lips_emb = not args.word_level)
    elif args.model_name == 'ConvLips':
        net = ConvLipsModel(n_char+1, args.embed_size, args.hidden_size, args.n_classes, kernel_size = args.kernel_size, n_layers = args.n_layers,p=args.p, reduce=args.reduce, lips_emb = not args.word_level, batch_norm=args.batch_norm, activation = args.activation)
    else:
        raise ValueError('Model name not recognized')

    '''
    use pretrained embeddings
    '''
    if args.word_level:
        net.emb.weight.data = embeddings.float()
        net.emb.weight.requires_grad = False
        net.char_to_id = word_to_id
    else:
        net.char_to_id = char_to_id

    os.makedirs('results_' + args.model_name, exist_ok=True)
    
    if not args.train_verifiable:
        output_folder = 'results_' + args.model_name + ('_p' + str(args.p) if (args.p != 'inf' or args.model_name == 'Conv') else '') + ('word' if args.word_level else '') + '/' + args.output_folder + '_' + args.dataset + f'_valid{args.valid_size}_opt' + args.optimizer + '_sch' + args.scheduler + '_reduce-' + args.reduce + f'_nl{args.n_layers}_bs{args.batch_size}_es{args.embed_size}_hs{args.hidden_size}_ks{args.kernel_size}_lr{args.lr}_lm{args.lm_weight}_mp{args.masking_prob}_lips_reg{args.lips_reg}_steps_dec{args.steps_decrease}'+ ('_act' + args.activation if args.activation != 'ReLU' else '') + f'_seed{args.seed}'
    else:
        output_folder = 'results_' + args.model_name + ('_p' + str(args.p) if args.p != 'inf' else '') + '/' + args.output_folder + '_' + args.dataset + f'_valid{args.valid_size}_opt' + args.optimizer + '_sch' + args.scheduler + 'verifiable_reduce-' + args.reduce + f'_nl{args.n_layers}_bs{args.batch_size}_es{args.embed_size}_hs{args.hidden_size}_ks{args.kernel_size}_lr{args.lr}_steps_dec{args.steps_decrease}_seed{args.seed}'
    os.makedirs(output_folder, exist_ok=True)

    # # define the optimizer.
    lr_steps = args.epochs * (len(train_loader.dataset) // args.batch_size + 1)
    if args.optimizer == 'SGD':
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    #scheduler = optim.lr_scheduler.CyclicLR(opt, base_lr=0, max_lr=args.lr, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    if args.scheduler == 'constant':
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.steps_decrease, gamma=0.1)
    elif args.scheduler == 'cyclic':
        lr_steps = args.epochs*len(train_loader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0, max_lr=args.lr,
                                                                  step_size_up=lr_steps / 2,
                                                                  step_size_down=lr_steps / 2, cycle_momentum=False)
    # define device (cuda if possible)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Using GPU?', cuda)
    # define minimization objective (for classification we use cross entropy loss)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion_chars = torch.nn.CrossEntropyLoss(reduce='none').to(device)

    net.to(device)

    reg_aux = args.lips_reg
    args.lips_reg = 0

    # # aggregate losses and accuracy.
    train_losses, acc_list = [], []
    df = {'epoch': [], 'train_loss': [],  'train_lips': [],'train_acc': [], 'valid_acc': [], 'valid_verified_1':[], 'valid_verified_2':[], 'valid_verified_3':[], 'valid_oracle':[], 'valid_oracle_replace-only':[],'valid_ibp':[],'valid_ibp_replace-only_1':[],'valid_ibp_replace-only_2':[],'valid_ibp_replace-only_3':[],'valid_ibp_1':[], 'test_acc': []}
    acc = 0.
    ver_acc_1 = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        if epoch>=0: args.lips_reg = reg_aux
        df['epoch'].append(epoch)
        print('Epoch {} (previous validation accuracy: {:.03f}, verified {:.03f})'.format(epoch, acc, ver_acc_1))
        loss_tr, lips_tr = train(train_loader, net, opt, criterion,criterion_chars, epoch, device,grad_clip=args.grad_clip, lm_weight=args.lm_weight, masking_prob = args.masking_prob, lips_reg=args.lips_reg, scheduler=(scheduler if args.scheduler =='cyclic' else None))
            
        train_acc = test(net, train_loader, device=device)
        if args.valid_size == 0:
            acc = test(net, test_loader, device=device)
            ver_acc_1,ver_acc_2,ver_acc_3 = verify(net, (char_to_id if not args.word_level else word_to_id), test_loader.dataset,device, output_folder=output_folder,pad=pad, synonyms=synonyms, tokenizer = (char_tokenizer if not args.word_level else word_tokenizer))
        else:
            acc = test(net, valid_loader, device=device)
            ver_acc_1,ver_acc_2,ver_acc_3 = verify(net, (char_to_id if not args.word_level else word_to_id), valid_loader.dataset,device, output_folder=output_folder,pad=pad, synonyms=synonyms, tokenizer = (char_tokenizer if not args.word_level else word_tokenizer))

        if args.eval_ibp:
            ver_acc_ibp_replace_only_1 = verified_acc_ibp_replace(net,  char_to_id,  valid_loader.dataset, device, pad=pad,delta=1)
            ver_acc_ibp_replace_only_2 = None
            ver_acc_ibp_replace_only_3 = None
            ver_acc_ibp_1 = verified_acc_ibp(net,  char_to_id,  valid_loader.dataset, device, pad=pad)
            ver_acc_ibp = None
        else:
            ver_acc_ibp_replace_only_1 = None
            ver_acc_ibp_replace_only_2 = None
            ver_acc_ibp_replace_only_3 = None
            ver_acc_ibp_1 = None
            ver_acc_ibp = None

        if args.eval_oracle:
            ver_acc_brute_replace_only = verified_acc_bruteforce(net, char_to_id,  valid_loader.dataset, device, replace = True, insert = False, delete=False,pad=pad)
            ver_acc_brute = verified_acc_bruteforce(net, char_to_id,  valid_loader.dataset, device, replace = True, insert = True, delete=True,pad=pad)
        else:
            ver_acc_brute_replace_only = None
            ver_acc_brute = None
        if args.scheduler == 'constant':
            scheduler.step()
        # Save weights if accuracy is improved
        if acc > best_acc:
            torch.save(net, os.path.join(output_folder, 'weights_best.pt'))
            torch.save(net.state_dict(), os.path.join(output_folder, 'weights_best_dict.pt'))
            best_acc = acc
            test_acc = test(net, test_loader, device=device)
            df['test_acc'].append(test_acc)
        else:
            df['test_acc'].append(None)
        torch.save(net, os.path.join(output_folder, 'weights_last.pt'))
        torch.save(net.state_dict(), os.path.join(output_folder, 'weights_last_dict.pt'))

        train_losses.append(loss_tr)
        acc_list.append(acc)
        df['train_loss'].append(loss_tr)
        df['train_lips'].append(lips_tr)
        df['train_acc'].append(train_acc)
        df['valid_acc'].append(acc)
        df['valid_verified_1'].append(ver_acc_1)
        df['valid_verified_2'].append(ver_acc_2)
        df['valid_verified_3'].append(ver_acc_3)
        df['valid_oracle'].append(ver_acc_brute)
        df['valid_oracle_replace-only'].append(ver_acc_brute_replace_only)
        df['valid_ibp'].append(ver_acc_ibp)
        df['valid_ibp_replace-only_1'].append(ver_acc_ibp_replace_only_1)
        df['valid_ibp_replace-only_2'].append(ver_acc_ibp_replace_only_2)
        df['valid_ibp_replace-only_3'].append(ver_acc_ibp_replace_only_3)
        df['valid_ibp_1'].append(ver_acc_ibp_1)
        pd_df = pd.DataFrame.from_dict(df, orient='columns')
        pd_df.to_csv(os.path.join(output_folder, f'log.csv'), index=False)