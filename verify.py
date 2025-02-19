import os
import torch
import argparse
import pandas as pd
from utils import get_dataloaders
from verification import verify, verified_acc_bruteforce, verified_acc_ibp, attack_charmer, verify_rsdel

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of a shallow transformer')
    parser.add_argument('--dataset', default='ag_news', type=str, help='glue datase or numbers_letters')
    parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--model_path', default='results/example_ag_news/weights_last.pt', type=str, help='default model weights')
    parser.add_argument('--n_samples', default=-1, type=int, help='number of samples to consider')

    args = parser.parse_args()

    

    if args.dataset in ['ag_news', 'imdb', 'fake-news']:
        pad = 1000
        pad_verif = 1010
    else:
        pad = 286
        pad_verif = 296

    out_folder = os.path.abspath(os.path.join(args.model_path, os.pardir))
    print(out_folder)

    char_to_id, train_loader, valid_loader, test_loader = get_dataloaders(args.dataset,16,pad = pad, valid_size=0)
    # create the model.
    n_char=len(char_to_id.keys())

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Using GPU?', cuda)

    if 'sentence' in train_loader.dataset[0]:
        key = 'sentence'
    else:
        key = 'text'
    
    net = torch.load(args.model_path, map_location=device)
    if not hasattr(net,'lips_act'):
        net.lips_act = 1

    if not hasattr(net,'lips_emb'):
        net.lips_emb = True
        net.batch_norm = False

    if args.dataset != 'numbers_letters':
        if args.n_samples == -1:
            l = len(test_loader.dataset[key])
        else:
            l = args.n_samples
        if args.dataset == 'fake-news':
            cut_dataset = [{key: str(train_loader.dataset[-i][key])[:1000] if len(str(train_loader.dataset[-i][key]))>1000 else str(train_loader.dataset[-i][key]) , 'label': train_loader.dataset[-i]['label']} for i in range(l)]
        else:
            cut_dataset = [{key: test_loader.dataset[key][i][:1000] if len(test_loader.dataset[key][i])>1000 else test_loader.dataset[key][i], 'label': test_loader.dataset['label'][i]} for i in range(l)]
    else:
        char_to_id = net.char_to_id
        cut_dataset = test_loader.dataset

    if os.path.isfile(os.path.join(out_folder, '_results_bruteforce.csv')):
        df = pd.read_csv(os.path.join(out_folder, '_results_bruteforce.csv'))
        if len(df)< args.n_samples:
            print('bf',verified_acc_bruteforce(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif, resume=True))
    else:
        print('bf',verified_acc_bruteforce(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif))
    
    if os.path.isfile(os.path.join(out_folder, '_results_ibp.csv')):
        df = pd.read_csv(os.path.join(out_folder, '_results_ibp.csv'))
        if len(df)< args.n_samples:
            print('ibp',verified_acc_ibp(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif, resume=True))
    else:
        print('ibp',verified_acc_ibp(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif))

    if os.path.isfile(os.path.join(out_folder, '_results_charmer.csv')):
        df = pd.read_csv(os.path.join(out_folder, '_results_charmer.csv'))
        if len(df)< args.n_samples:
            print('charmer',attack_charmer(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif, resume=True))
    else:
        print('charmer',attack_charmer(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif))

    if os.path.isfile(os.path.join(out_folder, '_results_rsdel.csv')):
        df = pd.read_csv(os.path.join(out_folder, '_results_rsdel.csv'))
        if len(df)< args.n_samples:
            print('rsdel',verify_rsdel(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif, resume=True))
    else:
        print('rsdel',verify_rsdel(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif))

    #No need to resume ours!
    print('ours',verify(net, char_to_id, cut_dataset, device, output_folder = out_folder,pad = pad_verif))
