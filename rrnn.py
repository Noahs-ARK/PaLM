import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import xavier_uniform

def RRNN_Compute_CPU(d, k, bw=False):
    """CPU version of the core RRNN computation.

    Has the same interface as RRNN_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    def rrnn_compute_cpu(u, c_init=None):
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)
        if c_init is None:
            assert False
        else:
            c_init = c_init.contiguous().view(batch, d)

        u, forget = u[..., 0], u[..., 1]

        c_prev = c_init
        cs = []
        if bw:
            for t in range(length-1, -1, -1):
                c_t = c_prev * forget[t, ...] + u[t, ...]
                c_prev = c_t
                cs.append(c_t)
            cs.reverse()
            cs = torch.stack(cs, dim=1)
        else:
            for t in range(length):
                c_t = c_prev * forget[t, ...] + u[t, ...]
                c_prev = c_t
                cs.append(c_t)
            cs = torch.stack(cs, dim=0)
        c_final = c_t
        return cs, c_final

    return rrnn_compute_cpu


class RRNNCell(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 dropout=0.2,
                 rnn_dropout=0.2,
                 nl="tanh",
                 use_output_gate=True):
        super(RRNNCell, self).__init__()
        assert (n_out % 2) == 0
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.use_output_gate = use_output_gate  # borrowed from qrnn
        self.use_output_gate = False
        self.nl = nl
        assert self.nl in ["tanh", "relu", "none"]
        if self.nl == "tanh":
            self.nonlinearity = torch.tanh
        elif self.nl == "relu":
            self.nonlinearity = torch.relu
        else:
            self.nonlinearity = None
        self.nonlinearity = None

        # basic: in1, in2, f1, f2
        # optional: output.
        self.k = 3 if self.use_output_gate else 2
        self.n_bias = 3 if self.use_output_gate else 2
        self.size_per_dir = n_out*self.k

        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*self.n_bias
        ))
        self.init_weights()

    def init_weights(self):
        xavier_uniform(self.weight.data,
                       fan_in=self.n_in, fan_out=self.n_out,
                       gain=nn.init.calculate_gain(self.nl))
        # initialize bias
        self.bias.data.zero_()

    def forward(self, input, init_hidden=None, bw=False):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        length, batch  = input.size(0), input.size(1)
        if init_hidden is None:
            size = (batch, n_out)
            c_init = Variable(input.data.new(*size).zero_())
        else:
            c_init = init_hidden

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)

        weight_in = self.weight
        u_ = x_2d.mm(weight_in)
        u_ = u_.view(length, batch, n_out, self.k)
        bias = self.bias.view(self.n_bias, n_out)

        _, forget_bias = bias[:2, ...]
        if self.use_output_gate:
            output_bias = bias[2, ...]
            output = (u_[..., 2] + output_bias).sigmoid()

        u = Variable(u_.data.new(length, batch, n_out, 2))

        u[..., 1] = (u_[..., 1] + forget_bias).sigmoid()
        u[..., 0] = u_[..., 0] * (1. - u[..., 1])  # input 1

        if False and input.is_cuda:
            from rrnn_gpu import RRNN_Compute_GPU
            RRNN_Compute = RRNN_Compute_GPU(n_out, 2)
        else:
            RRNN_Compute = RRNN_Compute_CPU(n_out, 2, bw=bw)

        cs, c_final = RRNN_Compute(u, c_init)
        if self.use_output_gate and False:
            cs = output * cs
        if self.nonlinearity is not None and False:
            cs = self.nonlinearity(cs)
        return cs.view(length, batch, -1), c_final, u[..., 1]

    def get_dropout_mask_(self, size, p, rescale=True):
        w = self.weight.data
        if rescale:
            return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))
        else:
            return Variable(w.new(*size).bernoulli_(1-p))
