import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence


from .model import drop_sequence_sharedmask, _model_var
from .HighWayLSTM import Highway_Concat_BiLSTM
from .layer import NonLinear, Biaffine


class ImplicitDependencyRepresentations(nn.Module):
    def __init__(self, config, lstm_input_size, lstm_hidden_size, dep_label_space_size):
        super(ImplicitDependencyRepresentations, self).__init__()
        self.config = config
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dep_label_space_size = dep_label_space_size
        # softmax weights
        self.dep_gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.softmax_dep_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor([0.0]))
                                                     for _ in range(self.config.dep_num_lstm_layers)])
        self.cuda = True

        self.dep_bilstm = Highway_Concat_BiLSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.config.dep_num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )

        # dependency parsing module
        self.mlp_arc_dep = NonLinear(
            input_size=2 * config.lstm_hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size=2 * config.lstm_hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, self.dep_label_space_size,
                                     bias=(True, True))

    def init_masks(self, batch_size, lengths):
        max_sent_length = max(lengths)
        num_sentences = batch_size
        indices = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        masks = indices < lengths.unsqueeze(1)
        masks = masks.type(torch.FloatTensor)
        if self.cuda:
            masks = masks.cuda()
        return masks

    def forward(self, num_sentences, context_embeddings, sent_lengths, dep):
        masks = self.init_masks(num_sentences, torch.LongTensor(sent_lengths))
        lstm_out, _ = self.dep_bilstm(context_embeddings, masks)

        if self.training:
            lstm_out = drop_sequence_sharedmask(lstm_out, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(lstm_out)
        x_all_head = self.mlp_arc_head(lstm_out)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)

        self.arc_logits, self.rel_logits = arc_logit, rel_logit_cond

        heads, rels = dep[0], dep[1]
        loss = self.compute_dep_loss(heads, rels, sent_lengths.tolist())  # compute the dep loss
        return loss, self.arc_logits

    def compute_dep_loss(self, true_arcs, true_rels, lengths):
        b, l1, l2 = self.arc_logits.size()
        index_true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, padding_value=0, batch_first=True)
        )
        true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, padding_value=-1, batch_first=True)
        )

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-1000] * (l2 - length))
            mask = _model_var(self.parameters(), mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1, reduction="sum")

        size = self.rel_logits.size()
        output_logits = _model_var(self.parameters(), torch.zeros(size[0], size[1], size[3]))
        for batch_index, (logits, arcs) in enumerate(list(zip(self.rel_logits, index_true_arcs))):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.parameters(), pad_sequence(true_rels, padding_value=-1, batch_first=True))

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1, reduction="sum")

        loss = arc_loss + rel_loss
        return loss

    def get_reps(self, context_embeddings, masks):
        dep_lstm_out, dep_lstm_outputs = self.dep_bilstm.forward(context_embeddings, masks)
        normed_weights = F.softmax(torch.cat([param for param in self.softmax_dep_weights]), dim=0)
        normed_weights = torch.split(normed_weights, 1)  # split_size_or_sections=1, split_size=1)  # 0.3.0
        dep_representations = self.dep_gamma * \
                              sum([normed_weights[i] * dep_lstm_outputs[i] for i in
                                   range(self.config.dep_num_lstm_layers)])
        if self.training:
            lstm_out = drop_sequence_sharedmask(dep_lstm_out, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(dep_lstm_out)
        x_all_head = self.mlp_arc_head(dep_lstm_out)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)
        return dep_representations, arc_logit

