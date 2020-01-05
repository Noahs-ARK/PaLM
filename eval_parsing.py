import numpy as np
import torch
import data_ptb, data
from nltk import Tree
from utils import repackage_hidden, get_brackets
import argparser
args = argparser.args()
import model
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib


def right_branching(sent):
    def assemble_subtree(start, end):
        if end - start == 0:
            word = sent[start]
            return [word]

        left_trees = assemble_subtree(start, start)
        right_trees = assemble_subtree(start+1, end)
        children = left_trees + right_trees
        children = [Tree("NT", children)]
        return children

    tree = assemble_subtree(0, len(sent) - 2)
    return tree[0]

print('Args:', args)


def check_rb(sent, brackets):
    if len(sent) <= 2:
        return 0, 0
    brackets.remove((0, len(sent)-1))
    n = len(sent) - 1
    global d
    global rb_d
    d = 0
    rb_d = 0
    # print (brackets)
    def assemble_subtree(start, end):
        global d
        global rb_d
        if end - start <= 1:
            # if end - start == 1:
            #     d += 1
            #     rb_d += 1
            return
        split = -1
        for k in range(start+1, end):
            if (k, end+1) in brackets:
                split = k
                break
        
        if split == -1:
            for k in range(end, start, -1):
                if (start, k) in brackets:
                    split = k
                    break
        # print (start, end, split)
        if split == -1:
            return
        # if split == end:
        if split == start + 1:
            rb_d += 1
        d += 1
        assemble_subtree(start, split-1)
        assemble_subtree(split, end)
    assemble_subtree(0, n - 1)
    return d, rb_d


def parse_rb(sents, rps, gold_trees, batch_size=1):
    pred_trees = []
    prec_list, reca_list, f1_list = [], [], []
    n_instances = len(sents)
    dec, rb_dec = 0, 0
    match, pred, gold = 0., 0., 0.
    for i in range(n_instances):
        sent, rp = sents[i], rps[i].unsqueeze(0)
        if args.wsj10 and len(sent) > 41:
            continue
        if len(sent) > 41:
            continue
        # if len(sent) <= 3:
        #     continue
        print(" ".join(sent))
        pred_tree = right_branching(sent)
        pred_trees.extend(pred_tree)
        gold_tree = gold_trees[i]

        pred_brackets, _ = get_brackets([pred_tree])
        gold_brackets, _ = get_brackets([gold_tree])
        print(pred_tree)
        print(pred_brackets)
        print(gold_tree)
        print(gold_brackets)
        overlap = pred_brackets.intersection(gold_brackets)
        match += len(overlap)
        pred += len(pred_brackets)
        gold += len((gold_brackets))
        if len(gold_brackets) == 0 and len(pred_brackets) == 0:
            prec, reca, f1 = 1., 1., 1.
        elif len(overlap) == 0:
            prec, reca, f1 = 0., 0., 0.
        else:
            prec = float(len(overlap)) / len(pred_brackets)
            reca = float(len(overlap)) / len(gold_brackets)
            f1 = 2 * prec * reca / (prec + reca)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)
        d, rb_d = check_rb(sent, gold_brackets)
        dec += d
        rb_dec += rb_d
        print (f1)
        print("##########")
        print()
        print()
    prec_list, reca_list, f1_list \
        = np.array(prec_list).reshape((-1, 1)), np.array(reca_list).reshape((-1, 1)), np.array(
        f1_list).reshape((-1, 1))
    print('Mean Prec:', prec_list.mean(axis=0),
          ', Mean Reca:', reca_list.mean(axis=0),
          ', Mean F1:', f1_list.mean(axis=0))
    print (len(prec_list))
    prec = match / pred
    recall = match / gold
    f1 = 2 * prec * recall / (prec + recall)
    print (prec, recall, f1)
    print (rb_dec, dec)
    return pred_trees


def parse(model, sents, dictionary, rps, gold_trees, batch_size=1, wsj10=False, debug=False):
    model.eval()
    match, pred, gold = 0., 0., 0.
    dec, rb_dec = 0, 0
    nn = 0
    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        c_hidden = model.init_c_hidden(batch_size)
        pred_trees = []
        prec_list, reca_list, f1_list = [], [], []
        n_instances = len(sents)
        for i in range(n_instances):
            sent, rp = sents[i], rps[i].unsqueeze(0)
            if wsj10 and len(sent) > 41:
                continue
            if len(sent) > 41:
                continue
            # if len(sent) <= 3:
            #     continue
            nn = nn + len(sent) - 2
            assert len(sent) - 2 >= 0
            data = torch.LongTensor(dictionary.token2ids(sent)).unsqueeze(1)
            assert data.shape[0] == len(sent)
            if args.cuda:
                data = data.to(device=torch.device("cuda"))
                rp = rp.to(device=torch.device("cuda"))
            model.max_span_length = len(sent)
            model._att_._max_span_length_ = len(sent)
            # hidden = model.init_hidden(batch_size)
            c_hidden = model.init_c_hidden(batch_size)
            hidden = repackage_hidden(hidden)
            c_hidden = repackage_hidden(c_hidden)

            pred_tree, hidden, c_hidden_ = model.parse(data, sent, hidden, c_hidden, debug=debug)
            model._att_._max_span_length_ = args.max_span_length
            model.max_span_length = args.max_span_length
            pred_trees.extend(pred_tree)
            gold_tree = gold_trees[i]
            pred_brackets, _ = get_brackets([pred_tree])
            gold_brackets, _ = get_brackets([gold_tree])

            overlap = pred_brackets.intersection(gold_brackets)
            match += len(overlap)
            pred += len(pred_brackets)
            gold += len((gold_brackets))
            if len(gold_brackets) == 0 and len(pred_brackets) == 0:
                prec, reca, f1 = 1., 1., 1.
            elif len(overlap) == 0:
                prec, reca, f1 = 0., 0., 0.
            else:
                prec = float(len(overlap)) / len(pred_brackets)
                reca = float(len(overlap)) / len(gold_brackets)
                f1 = 2 * prec * reca / (prec + reca)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)
            d, rb_d = check_rb(sent, pred_brackets)
            dec += d
            rb_dec += rb_d

            if debug:
                print(" ".join(sent))
                print(pred_tree)
                print(pred_brackets)
                print(gold_tree)
                print(gold_brackets)
                print(f1)
                print("##########")
                print()
                print()
    prec_list, reca_list, f1_list \
        = np.array(prec_list).reshape((-1, 1)), np.array(reca_list).reshape((-1, 1)), np.array(
        f1_list).reshape((-1, 1))
    print('Mean Prec:', prec_list.mean(axis=0),
          ', Mean Reca:', reca_list.mean(axis=0),
          ', Mean F1:', f1_list.mean(axis=0))
    prec = match / pred
    recall = match / gold
    f1 = 2 * prec * recall / (prec + recall)
    print (prec, recall, f1)
    print (rb_dec, dec, nn)
    print (rb_dec * 1.0 / dec)
    return f1_list.mean(axis=0)

# test_ids = [corpus.token2ids(x) for x in corpus_ptb.test_sens]
if __name__ == "__main__":

    fn = 'corpus_10'
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data, max_span_length=args.max_span_length)
        torch.save(corpus, fn)

    fn_ptb = 'corpus_official_40'
    if os.path.exists(fn_ptb):
        print('Loading cached PTB dataset...')
        corpus_ptb = torch.load(fn_ptb)
    else:
        print('Producing PTB dataset...')
        corpus_ptb = data_ptb.Corpus(args.data_ptb)
        torch.save(corpus_ptb, fn_ptb)
    test_sents = corpus_ptb.train_sens if args.wsj10 else corpus_ptb.test_sens
    test_trees = corpus_ptb.train_trees if args.wsj10 else corpus_ptb.test_trees
    test_rps = corpus_ptb.train_rps if args.wsj10 else corpus_ptb.test_rps
    # model_load(args.save)
    # trees = parse_rb(sents=test_sents,  rps=test_rps, gold_trees=test_trees, batch_size=1)
    # dsa
    ntokens = len(corpus.dictionary)
    print(ntokens)
    model = model.RNNModel(ntoken=ntokens, args=args)
    if args.cuda:
        model = model.cuda()
    trees = parse(
        model=model,
        sents=test_sents,
        dictionary=corpus.dictionary,
        rps=test_rps,
        gold_trees=test_trees,
        batch_size=1, wsj10=args.wsj10, debug=False)

