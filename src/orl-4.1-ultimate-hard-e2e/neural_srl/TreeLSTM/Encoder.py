import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from .TreeGRU import DTTreeGRU, TDTreeGRU
from .Tree import creatTree


class EncoderRNN(nn.Module):
    """ The standard RNN encoder.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True)  # batch_first = False
        self.transform = nn.Linear(in_features=2 * hidden_size,
                                   out_features=input_size,
                                   bias=True)
        self.dt_tree = DTTreeGRU(input_size, hidden_size)
        self.td_tree = TDTreeGRU(input_size, hidden_size)

    def forward(self, input, heads, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns.
        inputs: [L, B, H], including the -ROOT-
        heads: [heads] * B
        """
        emb = self.dropout(input)

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None:
            outputs = unpack(outputs)[0]

        outputs = self.dropout(self.transform(outputs))
        max_length, batch_size, input_dim = outputs.size()
        trees = []
        indexes = np.full((max_length, batch_size), -1,
                          dtype=np.int32)  # a col is a sentence
        for b, head in enumerate(heads):
            root, tree = creatTree(
                head)  # head: a sentence's heads; sentence base
            root.traverse()  # traverse the tree
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree.forward(
            outputs, indexes, trees)
        td_outputs, td_hidden_ts = self.td_tree.forward(
            outputs, indexes, trees)

        outputs = torch.cat([dt_outputs, td_outputs], dim=2).transpose(0, 1)
        output_t = torch.cat([dt_hidden_ts, td_hidden_ts], dim=1).unsqueeze(0)

        return outputs, output_t
