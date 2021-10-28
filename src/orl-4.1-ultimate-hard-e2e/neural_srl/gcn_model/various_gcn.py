# from gcntree import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .tree import *


def inputs_to_tree_reps(head, words, maxlen, l, prune, pheads=None):
    head = head
    trees = [head_to_tree(head[i], None, l[i], prune, pheads=None) for i in range(len(l))]
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in
           trees]
    adj = np.concatenate(adj, axis=0)
    adj = torch.from_numpy(adj)
    return Variable(adj.cuda())


class HeterProbasedGraphConvLayerWRNN(nn.Module):
    """ HPGCN module operated on dependency graphs. """
    def __init__(self, config, input_dim, mem_dim, layers, hete_dep_num):
        super(HeterProbasedGraphConvLayerWRNN, self).__init__()
        # self.opt = opt
        self.config = config
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.2)
        self.hete_dep_num = hete_dep_num

        # rnn layer
        if self.config.gcn_rnn is True:
            input_size = input_dim
            self.rnn = nn.LSTM(input_size, self.config.gcn_rnn_hidden, self.config.gcn_rnn_layers, batch_first=True,
                               dropout=self.config.gcn_rnn_dropout, bidirectional=True)
            self.in_dim = self.config.gcn_rnn_hidden * 2
            self.rnn_drop = nn.Dropout(self.config.gcn_rnn_dropout)  # use on last layer output
            self.rnn_linear = nn.Linear(2 * self.config.gcn_rnn_hidden, self.mem_dim)
        # hp-dcgcn block
        self.hete_dep_module = nn.ModuleList()
        for j in range(self.hete_dep_num):
            weight_list = nn.ModuleList()
            for i in range(self.layers):
                weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))
            self.hete_dep_module.append(weight_list)
        # linear transformation
        self.linear_outputs = nn.ModuleList()
        for i in range(self.hete_dep_num):
            linear_output = nn.Linear(self.mem_dim, self.mem_dim)
            self.linear_outputs.append(linear_output)
        # linear combination
        self.linear_combination = nn.Linear(self.mem_dim * self.hete_dep_num, self.mem_dim)

    def reset_parameters(self):
        for single_gcn_module in self.hete_dep_module:
            for linear in single_gcn_module:
                nn.init.normal_(linear.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
                nn.init.constant_(linear.bias, 0)

        nn.init.normal_(self.linear_output.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
        nn.init.constant_(self.linear_output.bias, 0)
        nn.init.normal_(self.rnn_linear.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
        nn.init.constant_(self.rnn_linear.bias, 0)
        # linear combination
        nn.init.normal_(self.linear_combination.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
        nn.init.constant_(self.linear_combination.bias, 0)

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

    def forward(self, hete_heads, words, gcn_inputs, masks, maxlen=None):
        """generate hete adjs"""
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        hete_adjs = []
        for i, dep_heads in enumerate(hete_heads):
            heads, pheads = dep_heads
            adj = inputs_to_tree_reps(heads, words, maxlen, l, -1, pheads=pheads)
            hete_adjs.append(adj)

        batch_size = masks.size()[0]
        if self.config.gcn_rnn is True:
            gcn_inputs = self.rnn_linear.forward(self.rnn_drop(self.encode_with_rnn(gcn_inputs, masks, batch_size)))
        else:
            gcn_inputs = gcn_inputs
        final_outputs = []
        all_adjs = []
        for i in range(self.hete_dep_num):
            adj = hete_adjs[i]
            all_adjs.append(adj)
        adj = sum(all_adjs)
        gcn_module = self.hete_dep_module[0]
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = gcn_module[l](Ax)
            AxW = AxW + gcn_module[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_outputs[0].forward(gcn_outputs)
        out = self.gcn_drop(out)
        return out


class GraphConvLayerWRNN(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, config, input_dim, mem_dim, layers):
        super(GraphConvLayerWRNN, self).__init__()
        # self.opt = opt
        self.config = config
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(config.gcn_drop)
        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # rnn layer
        if self.config.gcn_rnn is True:
            input_size = input_dim
            self.rnn = nn.LSTM(input_size, self.config.gcn_rnn_hidden, self.config.gcn_rnn_layers, batch_first=True,
                               dropout=self.config.gcn_rnn_dropout, bidirectional=True)
            self.in_dim = self.config.gcn_rnn_hidden * 2
            self.rnn_drop = nn.Dropout(self.config.gcn_rnn_dropout)  # use on last layer output
            self.rnn_linear = nn.Linear(2 * self.config.gcn_rnn_hidden, self.mem_dim)
        else:
            self.rnn_drop = nn.Dropout(self.config.gcn_rnn_dropout)  # use on last layer output
            self.rnn_linear = nn.Linear(input_dim, self.mem_dim)
        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

    def reset_parameters(self):
        for linear in self.weight_list:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)

        nn.init.xavier_uniform_(self.linear_output.weight)
        nn.init.constant_(self.linear_output.bias, 0)
        nn.init.xavier_uniform_(self.rnn_linear.weight)
        nn.init.constant_(self.rnn_linear.bias, 0)

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

    def forward(self, adj, gcn_inputs, masks):
        batch_size = masks.size()[0]
        if self.config.gcn_rnn is True:
            gcn_inputs = self.rnn_linear.forward(self.rnn_drop(self.encode_with_rnn(gcn_inputs, masks, batch_size)))
        else:
            gcn_inputs = self.rnn_linear.forward(gcn_inputs)
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers):

        super(GraphConvLayer, self).__init__()
        # self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.2)
        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

    def reset_parameters(self):
        for linear in self.weight_list:
            nn.init.normal_(linear.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
            nn.init.constant_(linear.bias, 0)

        nn.init.normal_(self.linear_output.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))
        nn.init.constant_(self.linear_output.bias, 0)

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


class BaseGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers):

        super(BaseGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers

        # self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for _ in range(self.layers):
            self.weight_list.append(nn.Linear(mem_dim, mem_dim))

        # self.weight_list = self.weight_list.cuda()
        # self.linear_output = self.linear_output.cuda()

    def reset_parameters(self):

        for linear in self.weight_list:
            nn.init.normal(linear.weight, 0.0, 1.0 / (self.mem_dim ** 0.5))

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        # cache_list = [outputs]
        # output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            # AxW = AxW + self.weight_list[l](outputs)  # self loop
            # AxW = AxW / denom
            gAxW = F.relu(AxW)
            # cache_list.append(gAxW)
            # outputs = torch.cat(cache_list, dim=2)
            # output_list.append(self.gcn_drop(gAxW))
            outputs = gAxW

        # gcn_outputs = torch.cat(output_list, dim=2)
        # gcn_outputs = gcn_outputs + gcn_inputs
        # out = self.linear_output(gcn_outputs)
        return outputs
        # return out


class DirectionGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers):

        super(DirectionGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers

        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            # arc direction, the opposite arc direction, self-loop
            self.weight_list.extend([nn.Linear(mem_dim, mem_dim), \
                                     nn.Linear(mem_dim, mem_dim), nn.Linear(mem_dim, mem_dim)])

        # self.weight_list = self.weight_list.cuda()
        # self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        print('adj size: ', adj.size())
        adj_t = adj.transpose(1, 2)
        # denom = adj.sum(2).unsqueeze(2) #+ 1
        outputs = gcn_inputs
        # cache_list = [outputs]
        # output_list = []

        for l in range(self.layers):
            # arc direction
            Ax = adj.bmm(outputs)
            arc_direction_AxW = self.weight_list[l * 3](Ax)
            # the opposite arc direction
            Ax = adj_t.bmm(outputs)
            opposite_arc_direction_AxW = self.weight_list[l * 3 + 1](Ax)
            # self loop
            self_loop_AxW = self.weight_list[l * 3 + 2](Ax)
            AxW = arc_direction_AxW + opposite_arc_direction_AxW + self_loop_AxW
            gAxW = F.relu(AxW)
            outputs = gAxW

        # gcn_outputs = torch.cat(output_list, dim=2)
        # gcn_outputs = gcn_outputs + gcn_inputs
        # out = self.linear_output(gcn_outputs)
        return outputs


def head_to_tree(head, tokens, len_, prune, pheads=None):
    """
    Convert a sequence of head indexes into a tree object.
    """
    # tokens = tokens[:len_].tolist()
    head = head[:len_]  # no effect
    assert -1 in head
    root = None

    assert prune < 0
    # print("head", head)
    if prune < 0:
        nodes = [Tree() for _ in head]
        for i in range(len(nodes)):
            h = head[i] + 1
            if pheads is not None:
                nodes[i].phead = float(pheads[i])
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        assert False
    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            if c.phead > 0:
                ret[t.idx, c.idx] = c.phead  # 1 original
            else:
                ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

