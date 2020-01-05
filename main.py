import time
import math
import numpy as np
import torch
import sys
import data, data_ptb
import model
from utils import batchify, get_batch, repackage_hidden
import argparser
from eval_parsing import parse

args = argparser.args()
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


import os
import hashlib

fn = 'corpus_{}'.format(args.max_span_length)
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data, max_span_length=args.max_span_length)
    torch.save(corpus, fn)

fn_ptb = 'corpus_official_40'
if  os.path.exists(fn_ptb):
    print('Loading cached PTB dataset...')
    corpus_ptb = torch.load(fn_ptb)
else:
    print('Producing PTB dataset...')
    corpus_ptb = data_ptb.Corpus(args.data_ptb)
    torch.save(corpus_ptb, fn_ptb)

sys.stdout.flush()
eval_batch_size = 10
test_batch_size = 1

train_data, train_rps, train_trees = batchify(corpus.train, args.batch_size, args, corpus.train_trees)
val_data, val_rps, _ = batchify(corpus.valid, corpus.valid_rps, eval_batch_size, args)
test_data, test_rps, _ = batchify(corpus.test, corpus.test_rps, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss

criterion = None

ntokens = len(corpus.dictionary)
print(ntokens)
model = model.RNNModel(ntoken=ntokens, args=args)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = model.dropouti, model.dropouth, model.dropout, model.dropoute
    if model.wdrop:
        from weight_drop import WeightDrop

        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = model.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = model.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###

params, parser_params = [], []
for n, p in model.named_parameters():
    if "_att_" in n:
        parser_params.append(p)
    else:
        params.append(p)
for n, p in criterion.named_parameters():
    if "_att_" in n:
        parser_params.append(p)
    else:
        params.append(p)
params_to_clip = list(filter(lambda p: p.shape[0] != args.max_span_length-1, params))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Args:', args)
print('Model total parameters:', total_params)
sys.stdout.flush()
###############################################################################
# Training code
###############################################################################

def evaluate(data_source, rps, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        c_hidden = model.init_c_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, rp, _, targets = get_batch(
                data_source, i, args=args)
            hidden = repackage_hidden(hidden)
            c_hidden = repackage_hidden(c_hidden)
            output, _, hidden, c_hidden = model(data, hidden, c_hidden)
            total_loss += len(data) * criterion(
                model.decoder.weight, model.decoder.bias, output, targets).data
        return total_loss.item() / len(data_source)

def eval_parsing():
    test_sents = corpus_ptb.train_sens if args.wsj10 else corpus_ptb.test_sens
    test_trees = corpus_ptb.train_trees if args.wsj10 else corpus_ptb.test_trees
    test_rps = corpus_ptb.train_rps if args.wsj10 else corpus_ptb.test_rps
    f1 = parse(model=model, sents=test_sents, dictionary=corpus.dictionary, rps=test_rps, gold_trees=test_trees, batch_size=1, wsj10=args.wsj10)
    return f1

def train(update_parser=True):
    total_loss = 0
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)
    c_hidden = model.init_c_hidden(args.batch_size)
    
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, rp, tree, targets = get_batch(
            train_data, i, args=args, seq_len=seq_len, trees=train_trees)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        c_hidden = repackage_hidden(c_hidden)
        optimizer.zero_grad()
        if update_parser:
            parser_optimizer.zero_grad()
        output, span_dist, hidden, c_hidden, rnn_hs, dropped_rnn_hs \
            = model(data, hidden, c_hidden, return_h=True)

        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(
            args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(
            args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        
        optimizer.step()

        if update_parser:
            torch.nn.utils.clip_grad_norm_(parser_params, 1.0)
            parser_optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
best_f1 = -1.
# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
lr_reduced = False

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    parser_optimizer = torch.optim.Adam(parser_params, lr=1e-3, weight_decay=args.wdecay)
    model_save(args.save)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        update_parser = 't0' not in optimizer.param_groups[0]
        train(update_parser)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for n, prm in model.named_parameters():
                tmp[prm] = prm.data.clone()
                if "ax" in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data, val_rps, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            if val_loss2 < stored_loss:
                model_save(args.save + ".asgd")
                print('Saving Averaged!')
                stored_loss = val_loss2

            for n, prm in model.named_parameters():
                prm.data = tmp[prm].clone()

            if epoch == args.finetuning:
                model_load(args.save + ".asgd")
                val_loss = evaluate(val_data, val_rps, eval_batch_size)
                print('=' * 89)
                print('| Switching to finetuning | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('=' * 89)
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                best_val_loss = []

            if epoch > args.finetuning + 100 and len(best_val_loss) > args.nonmono and val_loss2 > min(
                    best_val_loss[:-args.nonmono]):
                model_load(args.save + ".asgd")
                test_loss = evaluate(test_data, test_rps, test_batch_size)
                print('=' * 89)
                print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
                    test_loss, math.exp(test_loss), test_loss / math.log(2)))
                print('=' * 89)
                print('Done!')

                sys.exit(1)
            best_val_loss.append(val_loss2)
        else:
            val_loss = evaluate(val_data, val_rps, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            if val_loss < stored_loss:
                model_save(args.save + ".sgd")
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                model_load(args.save + ".sgd")
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            best_val_loss.append(val_loss)
        print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_rps, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
