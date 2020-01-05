import torch
import math, nltk

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
EPS = 1e-6

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    input = input + EPS

    if size_average:
        return torch.mean(torch.sum(-target * torch.log(input), dim=2))
    else:
        return torch.sum(torch.sum(-target * torch.log(input), dim=2))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, args, trees=None):

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()

    if trees is not None:
        trees = trees / (torch.sum(trees, dim=-1, keepdim=True) + 1e-8)
        trees = trees.narrow(0, 0 , nbatch * bsz)
        trees = trees.view(-1, bsz, trees.size(-1)).contiguous()
        if args.cuda:
            trees = trees.cuda()
    return data, trees

def pad(sequences, pad_id):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(3, max(len(seq) for seq in sequences))
    return [ seq + [pad_id]*(max_len-len(seq)) for seq in sequences ]


def get_batch(source, i=0, args=None, seq_len=None, trees=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    tree = None
    if trees is not None:
        tree = trees[i:i+seq_len, :]
    target = source[i+1:i+1+seq_len].contiguous().view(-1)
    return data, tree, target

import torch as t
from typing import List
import itertools

def block_orthogonal(tensor: t.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        assert False, "tensor dimensions must be divisible by their respective " \
                      "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes)
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        data[block_slice] = t.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)



def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, fan_in=None, fan_out=None, gain=1):
    r"""Fills the input `Tensor` with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    if fan_in is None or fan_out is None:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with t.no_grad():
        return tensor.uniform_(-a, a)

def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            # print (node)
            # print (node_brac)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

