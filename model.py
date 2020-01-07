import math
import torch
import torch.nn as nn
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from span_parser import SpanScorer
from utils import block_orthogonal, xavier_uniform


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, args):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.drop = nn.Dropout(args.dropout)
        self.ninp = args.emsize
        self.encoder = nn.Embedding(ntoken, self.ninp)
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        self.dropout = args.dropout
        self.dropouti = args.dropouti
        self.dropouth = args.dropouth
        self.dropoute = args.dropoute
        self._max_span_length_ = args.max_span_length
        self.wdrop = args.wdrop
        self.tie_weights = args.tie_weights
        self.max_span_length = args.max_span_length
        self._cxt_size_ = args.cxtsize
        self._rrnn_size_ = args.rrnn_size
        self.nonlinearity = torch.tanh

        self.rnns = []
        for l in range(self.nlayers):
            in_size = self.ninp if l == 0 else self.nhid
            out_size = self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)
            self.rnns.append(torch.nn.LSTM(in_size, out_size, 1, dropout=0, batch_first=False))
        if self.wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.wdrop, variational=False) for rnn in
                         self.rnns]

        self._att_ = SpanScorer(input_size=self.nhid,
                                hidden_size=args.parser_size,
                                rrnn_size=self._rrnn_size_,
                                context_size=self._cxt_size_,
                                drop=self.dropouth,
                                max_span_length=self.max_span_length)
        self._hidden_layer_ = nn.Linear(self._cxt_size_, self.nhid, bias=True)
        self._hidden_gate_ = nn.Linear(self.nhid + self._cxt_size_, self.nhid, bias=True)

        self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(self.nhid, ntoken)
        if self.tie_weights:
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        initrange = (3.0 / self.ninp) ** 0.5
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        for rnn in self.rnns:
            for k, v in rnn.state_dict().items():
                if "weight" in k:
                    assert v.shape[0] % 4 == 0
                    block_orthogonal(v, [v.shape[0] // 4, v.shape[1]])
                elif "bias" in k:
                    v.data.zero_()

        xavier_uniform(self._hidden_layer_.weight.data)
        self._hidden_layer_.bias.data.fill_(0)
        xavier_uniform(self._hidden_gate_.weight.data,
                       gain=nn.init.calculate_gain("sigmoid"))
        self._hidden_gate_.bias.data.fill_(0)

    def forward(self, input, hidden, c_hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb_drop = self.lockdrop(emb, self.dropouti)
        rnn_h = emb_drop
        new_hidden = []
        rnn_hs, dropped_rnn_hs = [], []
        span_scores = None
        for l, rnn in enumerate(self.rnns):
            rnn_h, new_h = rnn(rnn_h, hidden[l])
            rnn_hs.append(rnn_h)
            raw_rnn_h = rnn_h
            new_hidden.append(new_h)
            if l != self.nlayers - 1:
                rnn_h = self.lockdrop(rnn_h, self.dropouth)
                dropped_rnn_hs.append(rnn_h)
            if l == self.nlayers - 2:
                span_scores, context, ch = self._att_(rnn_h, c_hidden)
                context = self.lockdrop(context, self.dropouth)
                feats = torch.cat([rnn_h, context], dim=2)
                gate = self._hidden_gate_(feats).sigmoid()
                context = self.nonlinearity(self._hidden_layer_(context))
                context = self.lockdrop(context, self.dropouth)
                rnn_h = raw_rnn_h * gate + context * (1. - gate)
                rnn_h = self.lockdrop(rnn_h, self.dropouth)

        output = self.lockdrop(rnn_h, self.dropout)
        dropped_rnn_hs.append(output)
        assert len(dropped_rnn_hs) == len(rnn_hs)
        result = output.view(output.size(0) * output.size(1), output.size(-1))
        if return_h:
            return result, span_scores, new_hidden, ch, rnn_hs, dropped_rnn_hs
        return result, span_scores, new_hidden, ch

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
            self.ninp if self.tie_weights else self.nhid)).zero_(),
                   weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                       self.ninp if self.tie_weights else self.nhid)).zero_())
                  for l in range(self.nlayers)]
        return hidden

    def init_c_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.max_span_length + 1, bsz, self.nhid)
 
    def parse(self, input, sent, hidden, c_hidden, debug=False):
        seq_len, batch_size = input.shape
        assert batch_size == 1

        hidden_len = c_hidden.shape[0]
        emb = embedded_dropout(self.encoder, input, dropout=0.)
        new_hidden = []
        rnn_h = emb
        for l, rnn in enumerate(self.rnns):
            rnn_h, new_h = rnn(rnn_h, hidden[l])
            new_hidden.append(new_h)
            if l == self.nlayers - 2:
                span_scores, context, ch = self._att_(rnn_h, c_hidden, eval=True)
                break

        trees = self._att_.parse(
            x=rnn_h,
            sent=sent,
            span_scores=span_scores,
            hidden_len=hidden_len,
            debug=debug)
        return trees, new_hidden, ch
