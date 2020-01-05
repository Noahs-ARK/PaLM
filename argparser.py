import argparse
import time
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def args():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--data_wiki', type=str, default='data/wikitext-2/',
                        help='location of the data corpus')
    parser.add_argument('--data_ptb', type=str, default='data/penn/',
                        help='location of the ptb data corpus')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--cxtsize', type=int, default=128,
                        help='size of context')
    parser.add_argument('--rrnn_size', type=int, default=128,
                        help='size of rrnn')
    parser.add_argument('--parser_size', type=int, default=64,
                        help='size of parser')
    parser.add_argument('--nhid', type=int, default=1100,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=4000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.2,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.6,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=15,
                        help='nonmono')
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--max_span_length', type=int, default=30,
                        help='max span length')
    parser.add_argument('--finetuning', type=int, default=500,
                        help='When (which epochs) to switch to finetuning')
    parser.add_argument('--wsj10', type=str2bool, default=False, help='WSJ10')
    args = parser.parse_args()
    args.tie_weights = True
    return args
