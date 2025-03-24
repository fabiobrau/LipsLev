import os
import sys
import time
from copy import copy
import torch
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import datasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

"""
Random
"""


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


"""
Perturbation sets
"""


def generate_sentence(S, z, u, V, k=1, alternative=None):
    """
    inputs:
    S: sentence that we want to modify
    z: location position
    u: selection character id
    V: vocabulary, list of UNICODE indices
    k: number of possible changes

    generate sentence with a single character modification at position z with character u
    """
    spaces = "".join(["_" for i in range(k)])
    xx = "".join([spaces + s for s in S] + [spaces])
    new_sentence = [c for c in xx]
    mask = []
    for i in range(len(S)):
        mask += [0 for i in range(k)] + [1]
    mask += [0 for i in range(k)]

    if type(z) == list:
        for p, c in zip(z, u):
            if V[c] != -1:
                new_sentence[p] = chr(V[c])
                mask[p] = 1
            else:
                new_sentence[p] = "_"
                mask[p] = 0
    else:
        if V[u] != -1:
            if (
                new_sentence[z] == chr(V[u])
                and (alternative is not None)
                and alternative != -1
            ):
                new_sentence[z] = chr(alternative)
                mask[z] = 1
            elif (
                new_sentence[z] == chr(V[u])
                and (alternative is not None)
                and alternative == -1
            ):
                new_sentence[z] = "_"
                mask[z] = 0
            else:
                new_sentence[z] = chr(V[u])
                mask[z] = 1
        else:
            new_sentence[z] = "_"
            mask[z] = 0

    new_sentence = [c if mask[i] else "" for i, c in enumerate(new_sentence)]
    new_sentence = "".join(new_sentence)
    return new_sentence


def generate_all_sentences_at_z(S, z, V, k=1, alternative=-1):
    """
    inputs:
    S: sentence that we want to modify
    z: location id
    V: vocabulary, list of UNICODE indices

    generates all the possible sentences by changing characters in the position z
    """
    return [
        generate_sentence(S, z, u, V, k, alternative=alternative) for u in range(len(V))
    ]


def generate_random_sentences_at_z(S, z, V, n, k=1, alternative=-1):
    """
    inputs:
    S: sentence that we want to modify
    z: location id
    V: vocabulary, list of UNICODE indices
    n: number of random samples

    generates all the possible sentences by changing characters in the position z
    """
    return [
        generate_sentence(S, z, u, V, k, alternative=alternative)
        for u in np.random.choice(range(len(V)), size=n, replace=(n > len(V)))
    ]


def generate_random_sentences(S, V, n, subset_z=None, k=1, alternative=None):
    """
    inputs:
    S: sentence that we want to modify
    V: vocabulary, list of UNICODE indices
    n: number of random samples to draw
    subset_z: subset of positions to consider (TODO: implement update for k>1)
    k: number of character modifications
    alternative: in the case len(V)=1, character to consider for switchings when the character to change is
    the one in the volcabulary


    generates n random sentences at distance k
    """
    if subset_z is None:
        subset_z = range(2 * len(S) + 1)

    out = [S for _ in range(n)]

    for _ in range(k):
        if k == 1:
            positions = np.random.choice(subset_z, size=n)
        else:
            positions = [
                np.random.choice(range(2 * len(s) + 1), size=1).item() for s in out
            ]
        replacements = np.random.choice(range(len(V)), size=n)

        out = [
            generate_sentence(s, z, u, V, k, alternative=alternative)
            for s, z, u in zip(out, positions, replacements)
        ]
    return out


def generate_all_sentences(S, V, subset_z=None, k=1, alternative=None):
    """
    inputs:
    S: sentence that we want to modify
    V: vocabulary, list of UNICODE indices
    subset_z: subset of positions to consider
    k: number of character modifications (TODO: k>1)
    alternative: in the case len(V)=1, character to consider for switchings when the character to change is
    the one in the volcabulary

    generates all the possible sentences by changing characters
    """
    out = []
    if subset_z is None:
        subset_z = range((k + 1) * len(S) + k)
    for z in subset_z:
        out += generate_all_sentences_at_z(S, z, V, k, alternative=alternative)
    return out


def get_all_inserts(sentence, char_to_id):
    out = []
    for c in char_to_id:
        s = c + sentence
        out.append(s)
    for p in range(1, len(sentence)):
        for c in char_to_id:
            s = sentence[:p] + c + sentence[p:]
            out.append(s)
    for c in char_to_id:
        s = sentence + c
        out.append(s)
    return out


def get_all_deletes(sentence, char_to_id):
    out = [sentence[1:], sentence[:-1]]
    for p in range(1, len(sentence) - 1):
        s = sentence[:p] + sentence[p + 1 :]
        out.append(s)
    return out


def get_all_replacements(sentence, char_to_id):
    out = [sentence]
    for p in range(len(sentence)):
        for c in char_to_id:
            if sentence[p] != c:
                if p == 0:
                    s = c + sentence[1:]
                elif p == len(sentence) - 1:
                    s = sentence[:-1] + c
                else:
                    s = sentence[:p] + c + sentence[p + 1 :]
                out.append(s)
    return out


def join_sentence(list_of_words):
    if len(list_of_words):
        sentence = list_of_words[0]
        for word in list_of_words[1:]:
            sentence += " " + word
        return sentence
    else:
        return ""


def get_all_paraphrases(sentence, synonyms):
    words = sentence.split(" ")
    S = []
    for i, word in enumerate(words):
        if word in synonyms.keys():
            for syn in synonyms[word]:
                w2 = copy(words)
                w2[i] = syn
                S.append(join_sentence(w2))
    return S


"""
Dataloader related
"""


def char_tokenizer(S, to_id):
    l = []
    for c in S:
        if c.lower() in to_id.keys():
            l.append(to_id[c.lower()])
    return torch.tensor(l)


def word_tokenizer(sentence, to_id):
    tokens = sentence.split()
    ids = []
    for t in tokens:
        if t not in to_id.keys():
            ids.append(0)
        else:
            ids.append(to_id[t])
    return torch.tensor(ids).long()


def collate_fn(data, to_id, tokenizer=char_tokenizer, pad=286):
    """
    data: is a list of tuples with (example, label)
          where 'example' is a tensor of arbitrary shape
          and labels are scalars
    """
    if "sentence" in data[0]:
        key = "sentence"
    else:
        key = "text"
    SS = [str(d[key]) for d in data]
    labels = [d["label"] for d in data]

    unpadded = [
        tokenizer(S, to_id)[:pad] if len(S) > pad else tokenizer(S, to_id)[:pad]
        for S in SS
    ]
    unpadded[0] = torch.nn.ConstantPad1d((0, pad - unpadded[0].shape[0]), 0)(
        unpadded[0]
    )
    return pad_sequence(unpadded, batch_first=True), torch.tensor(labels).long(), SS


def str2float(x):
    """
    Parse float and fractions using argument parser.
    """
    if "/" in x:
        n, d = x.split("/")
        return float(n) / float(d)
    else:
        try:
            return float(x)
        except:
            raise argparse.ArgumentTypeError("Fraction or float value expected.")


def get_char_to_id(datasets):
    char_to_id = {}
    i = 1
    if "sentence" in datasets[0][0]:
        key = "sentence"
    else:
        key = "text"
    for dataset in datasets:
        for d in dataset:
            for c in d[key]:
                if c.lower() not in char_to_id.keys():
                    char_to_id[c.lower()] = i
                    i += 1
    return char_to_id


def get_char_to_id_list(datasets):
    char_to_id = {}
    i = 1
    if "sentence" in datasets[0].columns:
        key = "sentence"
    else:
        key = "text"
    for dataset in datasets:
        for _, d in dataset.iterrows():
            if str(d[key]) != "nan":
                for c in d[key]:
                    if c.lower() not in char_to_id.keys():
                        char_to_id[c.lower()] = i
                        i += 1
    return char_to_id


def generate_numbers_and_letters(n=5000, p_m=0.7, min_length=10, max_length=20):
    letters = [
        "q",
        "w",
        "e",
        "r",
        "t",
        "y",
        "u",
        "i",
        "o",
        "p",
        "a",
        "s",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "z",
        "x",
        "c",
        "v",
        "b",
        "n",
        "m",
    ]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    dataset = []

    for i in range(n):
        l = np.random.choice(range(min_length, max_length + 1))
        m = np.random.choice([0, 1])
        p = [0, 0]
        p[m] = p_m
        if m == 1:
            p[0] = 1 - p_m
        else:
            p[1] = 1 - p_m
        ids = np.random.choice([0, 1], p=p, size=l)
        sentence = []
        for j in range(l):
            if ids[j] == 1:
                sentence.append(np.random.choice(letters))
            else:
                sentence.append(np.random.choice(numbers))
        dataset.append({"sentence": "".join(sentence), "label": m})
    return dataset


def get_dataloaders(
    dataset,
    batch_size,
    word_to_id=None,
    word_level=False,
    n=13000,
    p_m=0.8,
    min_length=10,
    max_length=40,
    pad=286,
    valid_size=1000,
):
    if dataset == "numbers_letters":
        train_set = generate_numbers_and_letters(
            n, p_m, min_length=min_length, max_length=max_length
        )
        valid_set = generate_numbers_and_letters(
            200, p_m, min_length=min_length, max_length=max_length
        )
        test_set = generate_numbers_and_letters(
            200, p_m, min_length=min_length, max_length=max_length
        )

    else:
        if dataset in ["fake-news"]:
            train = pd.read_csv("fake-news/train.csv")
            test = pd.read_csv("fake-news/test.csv")
            train_set = []
            test_set = []
            valid_set = []
            for i in range(len(train) - 1000):
                train_set.append(
                    {"text": train["text"].iloc[i], "label": train["label"].iloc[i]}
                )
            for i in range(1000):
                test_set.append(
                    {"text": train["text"].iloc[-i], "label": train["label"].iloc[-i]}
                )
        elif dataset in ["ag_news", "imdb"]:
            dataset_dict = datasets.load_dataset(dataset)
            test_set = dataset_dict["test"]
            valid_set = []
            train_set = []
            # TODO fix the validation set ordering for imdb as it is ordered by label
            for i in range(valid_size):
                valid_set.append(
                    {
                        "text": dataset_dict["train"][-i]["text"],
                        "label": dataset_dict["train"][-i]["label"],
                    }
                )
            for i in range(len(dataset_dict["train"]) - valid_size):
                train_set.append(
                    {
                        "text": dataset_dict["train"][i]["text"],
                        "label": dataset_dict["train"][i]["label"],
                    }
                )
        elif dataset in ["sst2", "cola"]:
            dataset_dict = datasets.load_dataset("glue", dataset)
            test_set = dataset_dict["validation"]

            valid_set = []
            train_set = []
            for i in range(valid_size):
                valid_set.append(
                    {
                        "sentence": dataset_dict["train"][-i]["sentence"],
                        "label": dataset_dict["train"][-i]["label"],
                    }
                )
            for i in range(len(dataset_dict["train"]) - valid_size):
                train_set.append(
                    {
                        "sentence": dataset_dict["train"][i]["sentence"],
                        "label": dataset_dict["train"][i]["label"],
                    }
                )
    if not word_level:
        if dataset == "fake-news":
            char_to_id = get_char_to_id_list([train, test])
        elif dataset == "malware":
            from maltorch.src.maltorch.datasets.binary_dataset import BinaryDataset

            allset = BinaryDataset(
                goodware_directory="malwares/goodexe",
                malware_directory="malwares/malexe",
                max_len=1024 * 256,
            )
            sets = torch.utils.data.random_split(allset, [0.7, 0.1, 0.2])
            char_to_id = dict(zip(range(256), range(256)))

            def collate_fn(data):
                inputs = [d[0] for d in data]
                labels = [d[1] for d in data]
                return pad_sequence(inputs, True, 256), torch.tensor(labels).long()

            return char_to_id, *(
                DataLoader(d, batch_size, shuffle=True, collate_fn=collate_fn)
                for d in sets
            )

    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(
                x, word_to_id, tokenizer=word_tokenizer, pad=pad
            ),
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_fn(
                x, word_to_id, tokenizer=word_tokenizer, pad=pad
            ),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_fn(
                x, word_to_id, tokenizer=word_tokenizer, pad=pad
            ),
        )
        return word_to_id, train_loader, valid_loader, test_loader


"""
Train/Test related
"""


def train(
    train_loader,
    net,
    optimizer,
    criterion,
    criterion_chars,
    epoch,
    device,
    grad_clip=float("inf"),
    lm_weight=0,
    masking_prob=0,
    lips_reg=0,
    scheduler=None,
    tokenizer=None,
    pad=286,
):
    """Perform single epoch of the training."""
    net.train()
    reg = None
    train_loss, correct, total = 0, 0, 0
    correct_chars, total_chars = 0, 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description("Epoch: {}".format(epoch))
    for idx, data_dict in pbar:
        text = data_dict[0]
        label = data_dict[1].to(device)
        if tokenizer is not None:
            inputs = tokenizer(data_dict[2], padding=True, return_tensors="pt")[
                "input_ids"
            ].to(device)
            inputs = torch.nn.functional.pad(
                inputs, (0, pad - inputs.shape[1]), "constant", 0
            )
        else:
            inputs = text.to(device)

        optimizer.zero_grad()
        # compute output
        bs = inputs.shape[0]
        length = inputs.shape[1]
        mask = torch.rand(size=[bs, length, 1], device=device) < 1 - masking_prob
        pred, char_logits = net(inputs, mask)
        if lm_weight > 0:
            loss = (
                criterion(pred, label)
                + lm_weight
                * (
                    (mask == 0) * criterion_chars(char_logits.transpose(1, 2), inputs)
                ).mean()
            )
        else:
            loss = criterion(pred, label)
        reg = net.compute_lips_reg(inputs)
        loss += lips_reg * reg
        assert not torch.isnan(loss), "NaN loss."
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=grad_clip, norm_type="inf"
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()
        total_chars += bs * length
        if lm_weight > 0:
            _, predicted_chars = torch.max(char_logits.data, 2)
            correct_chars += predicted_chars.eq(inputs).cpu().sum()
        acc = float(correct) / total
        acc_chars = float(correct_chars) / total_chars
        m2 = "Loss: {:.04f}, Acc: {:.06f}, Acc chars: {:.06f}"
        pbar.set_description(m2.format(float(train_loss), acc, acc_chars))
    pbar.clear()
    try:
        return train_loss, reg.item()
    except:
        return train_loss, reg


def test(net, test_loader, device="cuda", tokenizer=None, pad=286):
    """Perform testing, i.e. run net on test_loader data
    and return the accuracy."""
    net.eval()
    correct, total = 0, 0
    for idx, data in enumerate(test_loader):
        sys.stdout.write("\r [%d/%d]" % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        text = data[0].to(device)
        label = data[1].to(device)
        if tokenizer is not None:
            inputs = tokenizer(data[2], padding=True, return_tensors="pt")[
                "input_ids"
            ].to(device)
            inputs = torch.nn.functional.pad(
                inputs, (0, pad - inputs.shape[1]), "constant", 0
            )
        else:
            inputs = text.to(device)
        with torch.no_grad():
            pred, char_logits = net(inputs)
        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    return correct / total


if __name__ == "__main__":
    s = "hello my friend"
    char_to_id = {}
    counter = 0
    for c in s:
        if not c in char_to_id:
            char_to_id[c] = counter
            counter += 1
    V = [-1] + [ord(c) for c in char_to_id.keys()]

    S = generate_all_sentences(s, V)
    SS = (
        get_all_inserts(s, char_to_id)
        + get_all_deletes(s, char_to_id)
        + get_all_replacements(s, char_to_id)
    )
    print(len(S), len(SS))
    print(len(set(S)), len(set(SS)))
    print(set(S).difference(set(SS)))
