import math
import numpy as np
import torch
import data
from torch.autograd import Variable
from utils import batchify, get_batch, repackage_hidden
import argparser
args = argparser.args()
from utils import Input

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

fn = 'corpus'
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)
eval_batch_size = 10
test_batch_size = 1
train_data, train_rps = batchify(corpus.train, corpus.train_rps, args.batch_size, args)
val_data, val_rps = batchify(corpus.valid, corpus.valid_rps, eval_batch_size, args)
test_data, test_rps = batchify(corpus.test, corpus.test_rps, test_batch_size, args)
print('Args:', args)

def evaluate(data_source, rps, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    
    criterion = torch.nn.CrossEntropyLoss()
    ntokens = len(corpus.dictionary)
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.shape[1] - 1, args.bptt):
            data, rp, targets = get_batch(
                data_source, rps, i, batch_size=batch_size, args=args, evaluation=True)
            input = Input(x=data, rp=rp)
            output, hidden = model(input, hidden)
            # total_loss += data.size(1) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            output = torch.nn.functional.linear(output, model.decoder.weight, bias=model.decoder.bias)
            output = torch.nn.functional.log_softmax(output, dim=-1)
            output_flat = output.view(-1, ntokens)
            total_loss += data.size(1) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss.item() / data_source.shape[1]
model_load(args.save)

test_loss = evaluate(val_data, val_rps, 10)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    val_loss, math.exp(test_loss), val_loss / math.log(2)))
print('=' * 89)
dsa

test_loss = evaluate(test_data, test_rps, 1)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
