"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..shared.constants import PAD_ID
import numpy as np


class GCN(nn.Module):
    def __init__(self, config, input_dim, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.layers = num_layers

        # rnn layer
        if self.config.gcn_rnn is True:
            input_size = self.input_dim
            self.rnn = nn.LSTM(input_size, self.config.gcn_rnn_hidden, self.config.gcn_rnn_layers, batch_first=True,
                               dropout=self.config.gcn_rnn_dropout, bidirectional=True)
            self.in_dim = self.config.gcn_rnn_hidden * 2
            self.rnn_drop = nn.Dropout(self.config.gcn_rnn_dropout)  # use on last layer output

        self.in_drop = nn.Dropout(self.config.gcn_input_dropout)
        self.gcn_drop = nn.Dropout(self.config.gcn_gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        self.layer_normalization = nn.ModuleList()

        for layer in range(self.layers):
            # input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(self.in_dim, self.in_dim))
            self.layer_normalization.append(LayerNormalization(self.in_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = masks.data.eq(1).long().sum(1).squeeze()
        h0, c0 = rnn_zero_state(batch_size, self.config.gcn_rnn_hidden, self.config.gcn_rnn_layers)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)

        rnn_inputs = rnn_inputs[perm_idx]
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)

        _, unperm_idx = perm_idx.sort(0)
        rnn_outputs = rnn_outputs[unperm_idx]
        return rnn_outputs

    def forward(self, adj, embs, masks):
        batch_size = masks.size()[0]
        embs = self.in_drop(embs)
        # rnn layer
        if self.config.gcn_rnn is True:
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, batch_size))
        else:
            gcn_inputs = embs

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # # zero out adj for ablation
        # if self.opt.get('no_adj', False):
        #     adj = torch.zeros_like(adj)

        for l in range(self.layers):
            # print(gcn_inputs.size(), adj.size())
            x = gcn_inputs
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW)
            self.layer_normalization[l].forward(gcn_inputs + x)

        return gcn_inputs, mask


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):  #
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)  # 1e-3 is ok, because variance and std.
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out
