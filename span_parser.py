import math
import torch as t
import torch.nn as nn
from torch.nn import Parameter
from nltk import Tree
from utils import xavier_uniform
from rrnn import RRNNCell
from locked_dropout import LockedDropout

def _vector_length_(seq_len, max_length):
    return seq_len * (max_length - 3)

def _ij2index_(start_idx, end_idx, max_len):
    if end_idx >= max_len:
        offset = max_len * (max_len - 1) // 2 + (end_idx - max_len) * (max_len - 1)
        i = end_idx - max_len + 1
        return offset + (start_idx - i)
    else:
        offset = end_idx * (end_idx - 1) // 2
        return offset + start_idx

class SpanScorer(nn.Module):
    """ A greedy span-based constituency parser. """

    def __init__(self,
                 input_size,
                 hidden_size,
                 rrnn_size,
                 context_size,
                 drop=0.,
                 max_span_length=10):
        super(SpanScorer, self).__init__()
        assert self._fw_rrnn_ or self._bw_rrnn_
        self._input_size_ = input_size
        self._hidden_size_ = hidden_size
        self._rrnn_size_ = rrnn_size
        self._context_size_ = context_size
        self._droprate_ = drop
        self._lockdrop_ = LockedDropout()
        self._max_span_length_ = max_span_length
        self._nonlinearity_ = t.tanh
        self._softmax_ = t.nn.Softmax(dim=2)

        size = self._rrnn_size_ // 2
        self._w_gate_ = Parameter(t.randn(self._rrnn_size_, self._input_size_))
        self._b_gate_ = Parameter(t.randn(self._rrnn_size_))
        xavier_uniform(self._w_gate_.data,
                       fan_in=self._input_size_,
                       fan_out=size,
                       gain=nn.init.calculate_gain("sigmoid"))
        self._b_gate_.data.zero_()

        self._w1_span_ = Parameter(t.randn(self._hidden_size_, self._rrnn_size_))
        self._w1_token_ = Parameter(t.randn(self._hidden_size_, self._input_size_))
        self._b1_ = Parameter(t.randn(self._hidden_size_))
        self._w2_ = Parameter(t.randn(1, self._hidden_size_))

        xavier_uniform(self._w1_span_.data,
                       fan_in=self._rrnn_size_ + self._input_size_,
                       fan_out=self._hidden_size_)
        xavier_uniform(self._w1_token_.data,
                       fan_in=self._rrnn_size_ + self._input_size_,
                       fan_out=self._hidden_size_)
        xavier_uniform(self._w2_.data,
                       fan_in=self._hidden_size_,
                       fan_out=1)
        self._b1_.data.zero_()

        size = self._rrnn_size_ // 2
        self.rrnn_fw = RRNNCell(
            n_in=self._input_size_,
            n_out=size,
            dropout=0.,
            rnn_dropout=0.,
            nl="tanh",
            use_output_gate=False)
        self.rrnn_bw = RRNNCell(
            n_in=self._input_size_,
            n_out=size,
            dropout=0.,
            rnn_dropout=0.,
            nl="tanh",
            use_output_gate=False)

    def _aggregate_forget(self, f, bw=False):
        log_f = t.log(f)
        total_len = f.shape[0]
        f_aggs = []

        if bw:
            local_f = log_f[-1, ...]
            f_aggs.append(local_f)
            for i in range(total_len - 2, -1, -1):
                local_f = local_f + log_f[i, ...]
                f_aggs.append(local_f)

            f_aggs.reverse()
            f_aggs = t.stack(f_aggs, dim=2)
        else:
            local_f = log_f[0, ...]
            f_aggs.append(local_f)
            for i in range(1, total_len):
                local_f = local_f + log_f[i, ...]
                f_aggs.append(local_f)
            f_aggs = t.stack(f_aggs, dim=2)
        return f_aggs

    def _compute_gate_(self, w, b, x):
        g = t.einsum("hi,lbi->bhl", (w, x.clone()))
        return t.sigmoid(g + b.unsqueeze(-1))

    def forward(self, x, init_x, eval=False):
        """
        einsum notations:
            - h: hidden_size
            - i: input_size
            - l: length
            - b: batch_size
        """
        # x: [seq_len, batch, dim]
        seq_len, batch_size = x.shape[:2]
        hidden_len = 0 if init_x is None else init_x.shape[0]
        if hidden_len is not None:
            assert hidden_len == self._max_span_length_ + 1
        x_aug = x if init_x is None else t.cat([init_x, x], dim=0)

        size = self._rrnn_size_

        rrnn_h_fw, _, rrnn_f_fw = self.rrnn_fw(
            x_aug, init_hidden=x.new_zeros(batch_size, size))

        f_fw_aggs = self._aggregate_forget(rrnn_f_fw, bw=False)
        rrnn_h_fw = self._lockdrop_(rrnn_h_fw, self._droprate_)

        span_reprs_fw = self._compute_span_repr_(
            rrnn_h_fw, f_fw_aggs, seq_len, hidden_len, bw=False)

        rrnn_h_bw, _, rrnn_f_bw = self.rrnn_bw(
            x_aug, init_hidden=x.new_zeros(batch_size, size), bw=True)
        f_bw_aggs = self._aggregate_forget(rrnn_f_bw, bw=True)
        rrnn_h_bw = self._lockdrop_(rrnn_h_bw, self._droprate_)
        span_reprs_bw = self._compute_span_repr_(
            rrnn_h_bw, f_bw_aggs, seq_len, hidden_len, bw=True)

        span_reprs = t.cat([span_reprs_fw, span_reprs_bw], dim=1)

        g = self._compute_gate_(self._w_gate_, self._b_gate_, x)
        span_reprs = span_reprs * g.unsqueeze(-1)
        span_reprs = self._nonlinearity_(span_reprs)
        span_scores = self._compute_span_scores_(
            x=x, reprs=span_reprs)
        init_x = None if init_x is None else x_aug[-hidden_len:, ...]
        if eval:
            return span_scores, None, init_x
        span_dist, context = self._span_attention_(span_reprs, span_scores)
        # span_dist, [batch, seq_len, vec_len]
        # context, [batch, seq_len, dim]
        return span_dist, context, init_x

    def _compute_span_repr_(self, rrnn, f_aggs, seq_len, hidden_len, bw=False):
        # [len, batch, dim]
        span_x = rrnn.permute(1, 2, 0)
        if bw:
            reprs = []
            for j in range(seq_len):
                end_idx = j + hidden_len
                start_idx = max(0, end_idx - self._max_span_length_)
                # [k+1, end_idx-1]
                log_f = f_aggs[..., start_idx+1:end_idx] - f_aggs[..., end_idx].unsqueeze(-1)
                repr = span_x[..., start_idx+1:end_idx]  - span_x[..., end_idx].unsqueeze(-1) * t.exp(log_f)
                reprs.append(repr)

            reprs = t.stack(reprs, dim=2)
        else:
            reprs = []
            for j in range(seq_len):
                end_idx = j + hidden_len
                start_idx = max(0, end_idx - self._max_span_length_)

                log_f = f_aggs[..., end_idx-1].unsqueeze(-1) - f_aggs[..., start_idx:end_idx-1]
                repr = - span_x[..., start_idx:end_idx-1] * t.exp(log_f)
                reprs.append(repr)
            reprs = t.stack(reprs, dim=2)
            reprs = reprs + span_x[..., hidden_len-1:-1].unsqueeze(-1)

        return reprs

    def _span_attention_(self, span_reprs, span_scores):
        span_dist = self._softmax_(span_scores)
        context = t.einsum("lbv,bhlv->lbh", (span_dist.clone(), span_reprs))
        return span_dist, context

    def _right_branching_scores_(self, x, seq_len):
        batch_size = x.size(1)
        scores = x.new(data=[range(10, 0, -1)]) * 0.1
        scores = scores.view(1, 1, -1).repeat(seq_len, batch_size, 1)
        return scores, None

    def _compute_span_scores_(self, x, reprs):
        token_x = t.einsum("hi,lbi->bhl", (self._w1_token_, x))
        reprs = t.einsum("hi,bilv->bhlv", (self._w1_span_, reprs.clone()))
        reprs = reprs + token_x.unsqueeze(-1)

        # [batch_size, dim, seq_len, max_len-1]
        reprs = self._nonlinearity_(reprs + self._b1_.view(1, self._hidden_size_, 1, 1))
        # [batch_size, seq_len, max_len-1]
        scores = t.einsum("oh,bhlv->lbv", (self._w2_, reprs.clone()))

        return scores

    def parse(self, x, sent, span_scores, hidden_len, debug=False):
        span_scores = span_scores.permute(1, 0, 2)
        seq_len, batch_size = x.shape[:2]
        assert batch_size == 1
        
        if debug:
            print (span_scores)
        span_scores = span_scores.cpu().data
        span_dict = {}
        for j in range(seq_len - 1, -1, -1):
            end_idx = j + hidden_len
            start_idx = max(0, end_idx - self._max_span_length_ + 1)
            for k in range(start_idx, end_idx):
                span = (k - self._max_span_length_-1, j-1)

                if span[0] < 0:
                    continue
                if span not in span_dict:
                    assert k - start_idx >= 0
                    span_dict[span] = span_scores[0, j, k - start_idx]
                else:
                    assert False
        tree = self.construct_tree(span_scores=span_dict, sentence=sent, debug=debug)
        return tree

    def construct_tree(self, span_scores, sentence, debug=False):
        def assemble_subtree(start, end, sent_end):
            if end == start:
                word = sentence[start]
                return [word]

            argmax_split = -1
            argmax_score = -1e4
            argmax_left_score, argmax_right_score = -1e4, -1e4

            for k in range(start, end):
                left_score = span_scores[(start, k)]
                right_score = span_scores[(k + 1, end)]
                score = right_score
                if debug:
                    print("enu: [{}, {}), [{}, {}): ".format(start, k + 1, k + 1, end + 1),
                          "{} = {} + {}".format(score, left_score, right_score))

                if score > argmax_score:
                    argmax_score = score
                    argmax_split = k
                    argmax_left_score = left_score
                    argmax_right_score = right_score

            if debug:
                print("argmax: [{}, {}), [{}, {}): ".format(start, argmax_split + 1, argmax_split + 1, end + 1),
                      "{} = {} + {}".format(argmax_score, argmax_left_score, argmax_right_score))
            left_trees = assemble_subtree(start, argmax_split, sent_end)
            right_trees = assemble_subtree(argmax_split + 1, end, sent_end)
            children = left_trees + right_trees
            children = [Tree("NT", children)]
            return children

        tree = assemble_subtree(0, len(sentence) - 2, len(sentence) - 2)
        return tree[0]
