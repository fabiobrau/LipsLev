import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from utils import (
    collate_fn,
    char_tokenizer,
    get_all_replacements,
    get_all_inserts,
    get_all_deletes,
)
from utils_attacks import attack_text_charmer_classification, sample_deletes

import scipy.stats as stats


def BinLCB(k, n, alpha):
    """Compute the lower confidence bound for a binomial proportion.

    Args:
        k (int): Number of successes.
        n (int): Number of trials.
        alpha (float): Confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        float: Lower confidence bound for p.
    """
    if k == 0:
        return 0.0  # LCB for 0 successes is 0
    else:
        return stats.beta.ppf(alpha, k, n - k + 1)


def dERP(a, b, p=2):
    if not len(a) and len(b):
        return torch.linalg.norm(b, ord=p, dim=1).sum()
    elif not len(b) and len(a):
        return torch.linalg.norm(a, ord=p, dim=1).sum()
    elif len(a) and len(b):
        x = torch.linalg.norm(a[0, :], ord=p) + dERP(a[1:, :], b, p=p)
        y = torch.linalg.norm(b[0, :], ord=p) + dERP(a, b[1:, :], p=p)
        z = torch.linalg.norm(a[0, :] - b[0, :], ord=p) + dERP(a[1:, :], b[1:, :], p=p)
        return min([x, y, z])
    else:
        return 0


def get_biggest_synonym_change(
    model, tokenizer, to_id, sentence, synonyms, device, p=2
):
    words = sentence.split(" ")

    m = 0
    for i, word in enumerate(words):
        embw = model.emb(tokenizer(word, to_id).to(device))
        if word in synonyms.keys():
            if len(synonyms[word]):
                for w in synonyms[word]:
                    tokens = tokenizer(w, to_id)
                    embs = model.emb(tokens.to(device))
                    m = max(m, dERP(embw, embs, p=p))
    return m


"""
Char Level
"""


def verified_acc_bruteforce(
    model,
    char_to_id,
    dataset,
    device,
    replace=True,
    insert=True,
    delete=True,
    output_folder="results_verif",
    output_name="",
    pad=286,
    resume=False,
):
    model.eval()
    V = [-1] + [ord(i) for i in char_to_id.keys() if len(i) == 1]
    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"
        if resume:
            df = pd.read_csv(
                os.path.join(output_folder, output_name + "_results_bruteforce.csv")
            ).to_dict(orient="list")
        else:
            df = {
                "sentence": [],
                "set_size": [],
                "set_size_replace": [],
                "true_label": [],
                "pred_label": [],
                "adv_example": [],
                "pred_adv": [],
                "radius": [],
                "time": [],
            }
        total = 0
        verified = 0
        os.makedirs(output_folder, exist_ok=True)
        for d in tqdm(dataset[len(df["sentence"]) :]):
            total += 1
            sentence = d[key]
            label = d["label"]
            df["sentence"].append(sentence)
            df["true_label"].append(label)
            if len(sentence) > pad:
                sentence = sentence[:pad]
            x = char_tokenizer(sentence, char_to_id).to(device)

            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)
            out, _ = model(x)
            pred_label = out[0].argmax()
            df["pred_label"].append(pred_label.item())
            if pred_label == label:
                start = time.time()
                S = (
                    get_all_replacements(sentence, char_to_id)
                    + get_all_inserts(sentence, char_to_id)
                    + get_all_deletes(sentence, char_to_id)
                )
                X = [{"sentence": s, "label": label} for s in S]
                data = torch.utils.data.DataLoader(
                    X,
                    batch_size=512,
                    shuffle=False,
                    collate_fn=lambda x: collate_fn(x, char_to_id, pad=pad),
                )
                ver = True
                for x in data:
                    s = x[0].to(device)
                    l = x[1].to(device)
                    pred, _ = model(s)
                    if torch.any(torch.argmax(pred, dim=1) != l):
                        ver = False
                        for i, v in enumerate(torch.argmax(pred, dim=1)):
                            if v.item() != label:
                                adv_example = x[2][i]
                                pred_adv = v.item()
                                break
                        df["adv_example"].append(adv_example)
                        df["pred_adv"].append(pred_adv)
                        break
                finish = time.time()
                if ver:
                    verified += 1
                    df["adv_example"].append(None)
                    df["pred_adv"].append(None)
                df["radius"].append(int(ver))
                df["time"].append(finish - start)
                df["set_size"].append(len(S))
                df["set_size_replace"].append(-1)
            else:
                df["adv_example"].append(None)
                df["pred_adv"].append(None)
                df["radius"].append(None)
                df["time"].append(None)
                df["set_size"].append(None)
                df["set_size_replace"].append(None)
            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(output_folder, output_name + "_results_bruteforce.csv"),
                index=False,
            )
        return verified / total


def verified_acc_ibp_replace_batched(
    model, char_to_id, dataset, device, pad=286, delta=1
):
    model.eval()
    total = 0
    verified = 0
    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"
        for d in tqdm(dataset):
            total += 1
            s = d[key]
            label = torch.tensor([d["label"]], device=device)
            S = get_all_replacements(s, char_to_id)
            x = char_tokenizer(s, char_to_id).to(device)
            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)
            X = torch.tensor(
                [
                    [char_to_id[c] for c in z.lower()]
                    + [0 for _ in range(pad - len(z))]
                    for z in S
                ],
                device=device,
            )
            l, u = model.forward_bounds_till_reduce(x, X, delta=delta)
            pred_verif = model.bounds_logits(l, u, label)
            ver = True
            for target in range(model.n_classes):
                if target != label.item():
                    if pred_verif[0, label.item()] - pred_verif[0, target] <= 0:
                        ver = False
                        break
            if ver:
                verified += 1
        return verified / total


def verified_acc_ibp_replace(
    model,
    char_to_id,
    dataset,
    device,
    output_folder="results_verif",
    output_name="",
    pad=286,
    delta=1,
):
    model.eval()
    df = {
        "sentence": [],
        "true_label": [],
        "pred_label": [],
        "margin_verified": [],
        "radius": [],
        "time": [],
    }
    total = 0
    verified = 0
    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"
        for d in tqdm(dataset):
            total += 1
            sentence = d[key]
            label = torch.tensor([d["label"]], device=device)
            df["sentence"].append(sentence)
            df["true_label"].append(label.item())
            if len(sentence) > pad:
                sentence = sentence[:pad]
            x = char_tokenizer(sentence, char_to_id).to(device)
            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)

            S = get_all_replacements(sentence, char_to_id)
            X = [{"sentence": s, "label": label.cpu()} for s in S]
            data = torch.utils.data.DataLoader(
                X,
                batch_size=512,
                shuffle=False,
                collate_fn=lambda x: collate_fn(x, char_to_id, pad=pad),
            )
            out, _ = model(x)
            pred_label = out[0].argmax()
            df["pred_label"].append(pred_label.item())
            if pred_label == label:
                start = time.time()
                ver = True
                l = None
                u = None
                for xx in data:
                    l_, u_ = model.forward_bounds_till_reduce(
                        x, xx[0].to(device), delta=delta
                    )
                    if l == None:
                        l = l_
                    else:
                        l = torch.cat([l, l_], dim=0).min(dim=0, keepdim=True)[0]

                    if u == None:
                        u = u_
                    else:
                        u = torch.cat([u, u_], dim=0).max(dim=0, keepdim=True)[0]
                pred_verif = model.bounds_logits(l, u, label)
                min_margin = float("inf")
                for target in range(model.n_classes):
                    if target != label.item():
                        min_margin = min(
                            min_margin,
                            pred_verif[0, label.item()] - pred_verif[0, target],
                        )
                        if pred_verif[0, label.item()] - pred_verif[0, target] <= 0:
                            ver = False
                            break
                finish = time.time()
                if ver:
                    verified += 1
                df["margin_verified"].append(min_margin.item())
                df["radius"].append(int(ver))
                df["time"].append(finish - start)
            else:
                df["margin_verified"].append(None)
                df["radius"].append(None)
                df["time"].append(None)
            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(
                    output_folder, output_name + f"_results_ibp_replace_{delta}.csv"
                ),
                index=False,
            )
        return verified / total


def verified_acc_ibp(
    model,
    char_to_id,
    dataset,
    device,
    output_folder="results_verif",
    output_name="",
    pad=286,
    resume=False,
):
    model.eval()
    if resume:
        df = pd.read_csv(
            os.path.join(output_folder, output_name + "_results_ibp.csv")
        ).to_dict(orient="list")
    else:
        df = {
            "sentence": [],
            "true_label": [],
            "pred_label": [],
            "margin_verified": [],
            "radius": [],
            "time": [],
        }
    total = 0
    verified = 0
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"
        for d in tqdm(dataset[len(df["sentence"]) :]):
            total += 1
            sentence = d[key]
            if len(sentence) > pad:
                sentence = sentence[:pad]
            label = torch.tensor([d["label"]], device=device)
            df["sentence"].append(sentence)
            df["true_label"].append(label.item())
            x = char_tokenizer(sentence, char_to_id).to(device)
            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)
            out, _ = model(x)
            pred_label = out[0].argmax()
            df["pred_label"].append(pred_label.item())
            if pred_label == label:
                start = time.time()
                S = (
                    get_all_replacements(sentence, char_to_id)
                    + get_all_inserts(sentence, char_to_id)
                    + get_all_deletes(sentence, char_to_id)
                )
                X = [{"sentence": s, "label": label.cpu()} for s in S]
                data = torch.utils.data.DataLoader(
                    X,
                    batch_size=512,
                    shuffle=False,
                    collate_fn=lambda z: collate_fn(z, char_to_id, pad=pad),
                )
                ver = True
                l = None
                u = None
                for xx in data:
                    l_, u_ = model.forward_bounds_till_reduce(
                        x, xx[0].to(device), delta=1
                    )
                    if l == None:
                        l = l_
                    else:
                        l = torch.cat([l, l_], dim=0).min(dim=0, keepdim=True)[0]

                    if u == None:
                        u = u_
                    else:
                        u = torch.cat([u, u_], dim=0).max(dim=0, keepdim=True)[0]
                pred_verif = model.bounds_logits(l, u, label)
                min_margin = float("inf")
                for target in range(model.n_classes):
                    if target != label.item():
                        min_margin = min(
                            min_margin,
                            pred_verif[0, label.item()] - pred_verif[0, target],
                        )
                        if pred_verif[0, label.item()] - pred_verif[0, target] <= 0:
                            ver = False
                            break
                finish = time.time()
                if ver:
                    verified += 1
                df["margin_verified"].append(min_margin.item())
                df["radius"].append(int(ver))
                df["time"].append(finish - start)
            else:
                df["margin_verified"].append(None)
                df["radius"].append(None)
                df["time"].append(None)
            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(output_folder, output_name + "_results_ibp.csv"),
                index=False,
            )
        return verified / total


def get_certified_radius_single(
    model,
    x,
    margin,
    target,
    label,
    synonyms=None,
    tokenizer=None,
    to_id=None,
    device=None,
):
    """
    Compute largest levenshtein distance for which we can verify robustness at the given sentence and label

    """
    if synonyms is None:
        return margin // model.compute_lips(x, target, label)
    else:
        ME = get_biggest_synonym_change(
            model, tokenizer, to_id, x, synonyms, device, p=model.p
        )
        print(ME)
        return margin // (ME * model.compute_lips(x, target, label))


def verify(
    model,
    to_id,
    dataset,
    device,
    output_folder="results_verif",
    output_name="",
    pad=286,
    max_n=1000,
    synonyms=None,
    tokenizer=char_tokenizer,
):
    """
    compute certified radius for every sentence in the given dataset
    """
    model.eval()

    os.makedirs(output_folder, exist_ok=True)
    if max_n == -1:
        max_n = len(dataset)

    df = {
        # "sentence": [],
        "true_label": [],
        "pred_label": [],
        "margin": [],
        "radius": [],
        "time": [],
    }
    total = 0
    verified_1 = 0
    verified_2 = 0
    verified_3 = 0
    with torch.no_grad():
        # if "sentence" in dataset[0]:
        #    key = "sentence"
        # else:
        #    key = "text"
        pbar = tqdm(enumerate(dataset))

        for i, (sentence, label) in pbar:
            if i > max_n:
                break
            total += 1
            # sentence = d[key]
            if len(sentence) > pad:
                sentence = sentence[:pad]
            # label = d["label"]
            # df["sentence"].append(sentence)
            df["true_label"].append(label)
            if "sentence" in dataset[0]:
                x = tokenizer(sentence, to_id=to_id).to(device)
            else:
                x = sentence.to(device)
            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)

            out, _ = model(x)
            pred_label = out[0].argmax()
            df["pred_label"].append(pred_label.item())
            if pred_label == label:
                start = time.time()
                rs = []
                ms = []
                for target in range(model.n_classes):
                    if target != label:
                        margin = (out[0, label] - out[0, target]).item()
                        ms.append(margin)
                        if synonyms is None:
                            r = get_certified_radius_single(
                                model, x, margin, target, label
                            ).item()
                        else:
                            r = get_certified_radius_single(
                                model,
                                sentence,
                                margin,
                                target,
                                label,
                                synonyms=synonyms,
                                tokenizer=tokenizer,
                                to_id=to_id,
                                device=device,
                            ).item()
                        rs.append(r)

                r = min(rs)
                margin = min(ms)
                finish = time.time()
                df["radius"].append(r)
                df["margin"].append(margin)
                df["time"].append(finish - start)
                radia = df["radius"]
                radia = [r for r in radia if r is not None]
                pbar.set_description(
                    f"Radius (80p): {np.percentile(radia, 80)}, Radius  (50p): {np.percentile(radia, 50)}"
                )

                if model.p == 1:
                    x = 1
                elif model.p == 2:
                    x = 1  # np.sqrt(2)
                elif model.p == float("inf"):
                    x = 1

                if r >= 3 * x:
                    verified_3 += 1
                if r >= 2 * x:
                    verified_2 += 1
                if r >= 1 * x:
                    verified_1 += 1
            else:
                df["margin"].append(None)
                df["radius"].append(None)
                df["time"].append(None)

            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(output_folder, output_name + "_results_lipslev.csv"),
                index=False,
            )
        return verified_1 / total, verified_2 / total, verified_3 / total


def attack_charmer(
    model,
    to_id,
    dataset,
    device,
    output_folder="results_verif",
    output_name="",
    pad=286,
    max_n=1000,
    synonyms=None,
    tokenizer=char_tokenizer,
    resume=False,
):
    """
    compute certified radius for every sentence in the given dataset

    to_id: dictionary that maps characters to indices
    """
    model.eval()
    V = [-1] + [ord(i) for i in to_id.keys() if len(i) == 1]

    os.makedirs(output_folder, exist_ok=True)
    if max_n == -1:
        max_n = len(dataset)

    if resume:
        df = pd.read_csv(
            os.path.join(output_folder, output_name + "_results_charmer.csv")
        ).to_dict(orient="list")
    else:
        df = {
            "sentence": [],
            "adv_sentence": [],
            "true_label": [],
            "pred_label": [],
            "pred_label_adv_1": [],
            "pred_label_adv_2": [],
            "time": [],
        }

    total = 0
    verified_1 = 0
    verified_2 = 0
    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"

        for i, d in tqdm(enumerate(dataset[len(df["sentence"]) :])):
            if i > max_n:
                break
            total += 1
            sentence = d[key]
            if len(sentence) > pad:
                sentence = sentence[:pad]
            label = d["label"]
            df["sentence"].append(sentence)
            df["true_label"].append(label)
            x = tokenizer(sentence, to_id=to_id).to(device)
            x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)

            out, _ = model(x)
            pred_label = out[0].argmax()
            pred = pred_label.item()
            df["pred_label"].append(pred)

            if pred_label == label:
                start = time.time()
                adv_2, dist = attack_text_charmer_classification(
                    model,
                    tokenizer,
                    to_id,
                    pad,
                    sentence,
                    label,
                    device,
                    n=20,
                    k=2,
                    V=V,
                    debug=False,
                )
                finish = time.time()
                df["adv_sentence"].append(adv_2)
                df["pred_label_adv_1"].append(label if dist > 1 else -1)
                x = tokenizer(str(adv_2), to_id=to_id).to(device)[:pad]
                x = torch.nn.ConstantPad1d((0, pad - x.shape[0]), 0)(x).unsqueeze(0)
                out, _ = model(x)
                pred_label = out[0].argmax()
                pred_adv_2 = pred_label.item()
                df["pred_label_adv_2"].append(pred_adv_2)
                df["time"].append(finish - start)

                if model.p == 1:
                    x = 1
                elif model.p == 2:
                    x = 1
                elif model.p == float("inf"):
                    x = 1

                if pred_adv_2 == label:
                    verified_2 += 1
                if dist > 1:
                    verified_1 += 1
            else:
                df["adv_sentence"].append(None)
                df["pred_label_adv_1"].append(None)
                df["pred_label_adv_2"].append(None)
                df["time"].append(None)

            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(output_folder, output_name + "_results_charmer.csv"),
                index=False,
            )
        return verified_1 / total, verified_2 / total


def verify_rsdel(
    model,
    to_id,
    dataset,
    device,
    output_folder="results_verif",
    output_name="",
    pad=286,
    max_n=1000,
    synonyms=None,
    tokenizer=char_tokenizer,
    resume=False,
    p_del=0.9,
    n_pred=1000,
    n_bnd=4000,
):
    """
    compute certified radius for every sentence in the given dataset

    to_id: dictionary that maps characters to indices
    """
    model.eval()
    V = [-1] + [ord(i) for i in to_id.keys() if len(i) == 1]

    os.makedirs(output_folder, exist_ok=True)
    if max_n == -1:
        max_n = len(dataset)

    if resume:
        df = pd.read_csv(
            os.path.join(output_folder, output_name + "_results_rsdel.csv")
        ).to_dict(orient="list")
    else:
        df = {
            "sentence": [],
            "true_label": [],
            "pred_label": [],
            "radius": [],
            "time": [],
        }

    total = 0
    verified_1 = 0
    verified_2 = 0
    with torch.no_grad():
        if "sentence" in dataset[0]:
            key = "sentence"
        else:
            key = "text"

        for i, d in tqdm(enumerate(dataset[len(df["sentence"]) :])):
            if i > max_n:
                break
            total += 1
            sentence = d[key]
            if len(sentence) > pad:
                sentence = sentence[:pad]
            label = d["label"]
            df["sentence"].append(sentence)
            df["true_label"].append(label)
            S = sample_deletes(sentence, n=n_pred, p_del=p_del)
            X = [{"sentence": s, "label": label} for s in S]
            data = torch.utils.data.DataLoader(
                X,
                batch_size=512,
                shuffle=False,
                collate_fn=lambda x: collate_fn(x, to_id, pad=pad),
            )
            outs = []
            for x in data:
                s = x[0].to(device).long()
                # l = x[1].to(device)
                out, _ = model(s)
                outs.append(out)
            out = torch.cat(outs, dim=0)

            mu = (
                F.one_hot(out.argmax(dim=1), num_classes=out.shape[-1])
                .float()
                .mean(dim=0)
            )
            pred_label = mu.argmax().item()
            df["pred_label"].append(pred_label)

            if pred_label == label:
                start = time.time()
                SS = sample_deletes(sentence, n=n_bnd, p_del=p_del)
                X = [{"sentence": s, "label": label} for s in SS]
                data = torch.utils.data.DataLoader(
                    X,
                    batch_size=512,
                    shuffle=False,
                    collate_fn=lambda x: collate_fn(x, to_id, pad=pad),
                )
                outs = []
                for x in data:
                    s = x[0].to(device).long()
                    out, _ = model(s)
                    outs.append(out)
                out = torch.cat(outs, dim=0)
                sum_correct = (
                    F.one_hot(out.argmax(dim=1), num_classes=out.shape[-1])
                    .float()
                    .sum(dim=0)[pred_label]
                    .item()
                )

                lcb = BinLCB(sum_correct, n_bnd, 0.05)
                nu = 1 / 2
                if lcb < 0:
                    # Abstain
                    r = 0
                else:
                    # Table 1 from the RS-Del paper with delitions, insertions and replacements
                    r = np.log(1 + nu - mu[pred_label].item()) / np.log(p_del)
                finish = time.time()
                df["radius"].append(r)
                df["time"].append(finish - start)
                if r >= 1:
                    verified_1 += 1
                if r >= 2:
                    verified_2 += 1
            else:
                df["radius"].append(0)
                df["time"].append(None)

            pd_df = pd.DataFrame.from_dict(df, orient="columns")
            pd_df.to_csv(
                os.path.join(output_folder, output_name + "_results_rsdel.csv"),
                index=False,
            )
        return verified_1 / total, verified_2 / total
