import torch
import torch.nn.functional as F
import numpy as np

from torch import nn, softmax, log_softmax
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
from .HighWayLSTM import Highway_Concat_BiLSTM
from .layer import MLPRepScorer, SpanClassifier
from ..gcn_model.various_gcn import GraphConvLayerWRNN, inputs_to_tree_reps
from .implicit_syntactic_representations import ImplicitDependencyRepresentations
from .pre_trained_language_model import BERT_input, BERT_model


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class BiLSTMTaggerModel(nn.Module):
    """ Constructs the network and builds the following Theano functions:
        - pred_function: Takes input and mask, returns prediction.
        - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
    """

    def __init__(self, data, config, gpu_id=""):
        super(BiLSTMTaggerModel, self).__init__()
        self.config = config

        self.dropout = float(config.dropout)  # 0.2
        self.lexical_dropout = float(self.config.lexical_dropout)
        self.lstm_type = config.lstm_cell
        self.lstm_hidden_size = int(config.lstm_hidden_size)  # SRL: 300
        self.cons_num_lstm_layers = int(config.cons_num_lstm_layers)  # SRL: 300
        self.num_lstm_layers = int(config.num_lstm_layers)  # SRL:
        self.max_grad_norm = float(config.max_grad_norm)
        self.use_gold_predicates = config.use_gold_predicates
        self.use_gold_arguments = config.use_gold_arguments
        print("USE_GOLD_PREDICATES ? ", self.use_gold_predicates)
        print("USE_GOLD_ARGUMENTS ? ", self.use_gold_arguments)

        self.word_embedding_shapes = data.word_embedding_shapes
        self.vocab_size = data.word_dict.size()
        self.char_size = data.char_dict.size()
        self.pos_size = data.pos_dict.size()
        self.label_space_size = data.label_dict.size()
        self.cons_label_size = data.cons_label_dict.size()
        self.dep_label_space_size = data.dep_label_dict.size()
        self.cuda_id = gpu_id
        # needed constituent spans
        self.needed_cons_labels = ["NP", "VP", "SBAR", "PP", "NML", "ADVP", "ADJP"]
        self.needed_cons_labels_idx = []
        for label in self.needed_cons_labels:
            self.needed_cons_labels_idx.append(data.cons_label_dict.get_index(label))

        # Initialize layers and parameters
        word_embedding_shape = self.word_embedding_shapes
        assert word_embedding_shape[0] == self.vocab_size
        self.word_embedding_dim = word_embedding_shape[1]  # get the embedding dim
        self.word_embedding = nn.Embedding(word_embedding_shape[0], self.word_embedding_dim, padding_idx=0)
        # character embedding
        self.char_embedding = nn.Embedding(self.char_size, self.config.char_emb_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.pos_size, self.config.pos_emb_size, padding_idx=0)

        self.word_embedding.weight.data.copy_(torch.from_numpy(data.word_embeddings))
        self.word_embedding.weight.requires_grad = False

        # char cnn layer
        # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1...
        self.char_cnns = nn.ModuleList(
            [nn.Conv1d(self.config.char_emb_size, self.config.output_channel, int(kernel_size),
                       stride=1, padding=0) for kernel_size in self.config.kernel_sizes])
        # constituent label embedding
        self.constituent_label_embedding = nn.Embedding(self.cons_label_size, self.config.cons_label_dim, padding_idx=0)
        self.isr_gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.isr_weights = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor([1.0])) for _ in range(self.cons_num_lstm_layers)]
        )
        # bert
        #
        self.bert_dim = int(config.bert_dim)
        self.lstm_input_size = self.word_embedding_dim + 3 * self.config.output_channel  # word emb dim + char cnn dim
        if self.config.use_bert:
            self.bert_input = BERT_input(self.config.bert_vocab_path, self.config.bert_path, 4, self.bert_dim)
            self.lstm_input_size += self.bert_dim
        self.cons_gcn = GraphConvLayerWRNN(
            config, self.lstm_input_size, config.gcn_hidden_dim,
            config.gcn_num_layers
        )
        self.dep_gcn = GraphConvLayerWRNN(
            config, self.lstm_input_size, config.gcn_hidden_dim,
            config.gcn_dep_num_layers
        )
        self.implicit_dependency_representation = ImplicitDependencyRepresentations(
            config, self.lstm_input_size, self.lstm_hidden_size, self.dep_label_space_size
        )
        # Initialize HighwayBiLSTM
        self.cons_bilstm = Highway_Concat_BiLSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.cons_num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )
        if config.mtl_cons:
            self.lstm_input_size += 2 * self.lstm_hidden_size
        if config.mtl_dep:
            self.lstm_input_size += 2 * self.lstm_hidden_size
        if config.use_cons_gcn:
            self.lstm_input_size += config.gcn_hidden_dim
        if config.use_dep_gcn:
            self.lstm_input_size += config.gcn_hidden_dim
        self.bilstm = Highway_Concat_BiLSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )
        # argument starts and ends
        self.dse_start_scorer = MLPRepScorer(
            2 * self.lstm_hidden_size, self.config.arg_start_size, 2, dropout=self.dropout
        )
        self.dse_end_scorer = MLPRepScorer(
            2 * self.lstm_hidden_size, self.config.arg_end_size, 2, dropout=self.dropout
        )
        # dse rep
        self.dse_reps_0 = nn.Linear(2 * self.lstm_hidden_size, self.config.argu_size, bias=False)
        self.dse_reps_drop_0 = nn.Dropout(self.dropout)
        # span width feature embedding
        # self.span_width_embedding = nn.Embedding(self.config.max_arg_width, self.config.span_width_feature_size)
        # self.context_projective_layer = nn.Linear(self.config.argu_size, self.config.num_attention_heads)
        self.span_emb_size = 2 * self.config.argu_size
        # span_rep
        self.dse_reps = nn.Linear(self.span_emb_size, self.config.argu_size_u)
        self.dse_reps_drop = nn.Dropout(self.dropout)
        if self.config.mtl_cons and self.config.use_cons_labels:
            self.dse_rep_dim = self.config.argu_size_u + self.config.cons_label_dim
        else:
            self.dse_rep_dim = self.config.argu_size_u
        # span scores
        self.dse_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.dse_rep_dim, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])
        self.dse_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.dse_unary_score_projection = nn.Linear(self.config.ffnn_size, 2)
        # # predicate rep

        # argument starts and ends
        self.arg_start_scorer = MLPRepScorer(
            2 * self.lstm_hidden_size, self.config.arg_start_size, 2, dropout=self.dropout
        )
        self.arg_end_scorer = MLPRepScorer(
            2 * self.lstm_hidden_size, self.config.arg_end_size, 2, dropout=self.dropout
        )
        # argument rep
        self.argu_reps_0 = nn.Linear(2 * self.lstm_hidden_size, self.config.argu_size, bias=False)
        self.argu_reps_drop_0 = nn.Dropout(self.dropout)
        # span width feature embedding
        # self.span_width_embedding = nn.Embedding(self.config.max_arg_width, self.config.span_width_feature_size)
        # self.context_projective_layer = nn.Linear(self.config.argu_size, self.config.num_attention_heads)
        self.span_emb_size = 2 * self.config.argu_size
        # span_rep
        self.argu_reps = nn.Linear(self.span_emb_size, self.config.argu_size_u)
        self.argu_reps_drop = nn.Dropout(self.dropout)
        if self.config.mtl_cons and self.config.use_cons_labels:
            self.argu_rep_dim = self.config.argu_size_u + self.config.cons_label_dim
        else:
            self.argu_rep_dim = self.config.argu_size_u
        # span scores
        self.arg_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.argu_rep_dim, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(self.config.ffnn_size, 2)

        # srl scores
        self.srl_unary_score_input_size = self.dse_rep_dim + self.argu_rep_dim
        self.srl_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.srl_unary_score_input_size, self.config.ffnn_size)
             if i == 0 else nn.Linear(self.config.ffnn_size, self.config.ffnn_size)
             for i in range(self.config.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.srl_unary_score_projection = nn.Linear(self.config.ffnn_size, self.label_space_size - 1)
        # self.srl_bilinear_scorer = Biaffine(2 * self.lstm_hidden_size, self.span_emb_size, self.label_space_size - 1)
        # self.softmax_srl_scores_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor([1.0])) for _ in range(3)])
        # constituent classifier
        self.cons_classifier = SpanClassifier(
            2 * self.lstm_hidden_size, self.config.arg_end_size, 2,
            self.span_emb_size, self.config.argu_size, drop=self.dropout,
            use_boundary=True
        )
        self.cons_reps = nn.Linear(self.span_emb_size, self.config.argu_size_u)
        self.cons_reps_drop = nn.Dropout(self.dropout)
        # span scores
        self.cons_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.config.argu_size_u, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])
        self.cons_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.cons_unary_score_projection = nn.Linear(self.config.ffnn_size, self.cons_label_size - 1)

        # self.pred_loss_weight = nn.Parameter(torch.FloatTensor([1.0]))
        # self.argu_loss_weight = nn.Parameter(torch.FloatTensor([1.0]))
        # self.srl_loss_weight = nn.Parameter(torch.FloatTensor([1.0]))
        self.focal_loss_alpha = float(self.config.fl_alpha)  # 0.25
        self.focal_loss_gamma = float(self.config.fl_gamma)  # 2

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.char_embedding.weight)
        init.xavier_uniform_(self.pos_embedding.weight)

        for layer in self.char_cnns:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.dse_reps_0.weight)
        # initializer_1d(self.argu_reps_0.bias, init.xavier_uniform_)

        # init.xavier_uniform_(self.span_width_embedding.weight)
        # init.xavier_uniform_(self.context_projective_layer.weight)
        # initializer_1d(self.context_projective_layer.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.dse_reps.weight)
        initializer_1d(self.dse_reps.bias, init.xavier_uniform_)
        for layer in self.dse_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.dse_unary_score_projection.weight)
        initializer_1d(self.dse_unary_score_projection.bias, init.xavier_uniform_)

        # init.xavier_uniform_(self.pred_reps.weight)
        # initializer_1d(self.pred_reps.bias, init.xavier_uniform_)
        # for layer in self.pred_unary_score_layers:
        #     init.xavier_uniform_(layer.weight)
        #     initializer_1d(layer.bias, init.xavier_uniform_)
        # init.xavier_uniform_(self.pred_unary_score_projection.weight)
        # initializer_1d(self.pred_unary_score_projection.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.argu_reps_0.weight)
        # initializer_1d(self.argu_reps_0.bias, init.xavier_uniform_)

        # init.xavier_uniform_(self.span_width_embedding.weight)
        # init.xavier_uniform_(self.context_projective_layer.weight)
        # initializer_1d(self.context_projective_layer.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.argu_reps.weight)
        initializer_1d(self.argu_reps.bias, init.xavier_uniform_)
        for layer in self.arg_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.arg_unary_score_projection.weight)
        initializer_1d(self.arg_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.srl_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.srl_unary_score_projection.weight)
        initializer_1d(self.srl_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.cons_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.cons_unary_score_projection.weight)
        initializer_1d(self.cons_unary_score_projection.bias, init.xavier_uniform_)
        return None

    def init_masks(self, batch_size, lengths):
        max_sent_length = max(lengths)
        num_sentences = batch_size
        indices = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        masks = indices < lengths.unsqueeze(1)
        masks = masks.type(torch.FloatTensor)
        if self.cuda_id:
            masks = masks.cuda()
        return masks

    def sequence_mask(self, sent_lengths, max_sent_length=None):
        batch_size, max_length = sent_lengths.size()[0], torch.max(sent_lengths)
        if max_sent_length is not None:
            max_length = max_sent_length
        indices = torch.arange(0, max_length).unsqueeze(0).expand(batch_size, -1)
        mask = indices < sent_lengths.unsqueeze(1).cpu()
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    def get_char_cnn_embeddings(self, chars):
        num_sentences, max_sentence_length = chars.size()[0], chars.size()[1]
        chars_embeddings = self.char_embedding(chars)

        chars_embeddings = chars_embeddings.view(num_sentences * max_sentence_length,
                                                 chars_embeddings.size()[2], chars_embeddings.size()[3])
        # [Batch_size, Input_size, Seq_len]
        chars_embeddings = chars_embeddings.transpose(1, 2)
        chars_output = []
        for i, cnn in enumerate(self.char_cnns):
            chars_cnn_embedding = torch.relu(cnn.forward(chars_embeddings))
            pooled_chars_cnn_emb, _ = chars_cnn_embedding.max(2)
            chars_output.append(pooled_chars_cnn_emb)
        chars_output_emb = torch.cat(chars_output, 1)
        return chars_output_emb.view(num_sentences, max_sentence_length, chars_output_emb.size()[1])

    @staticmethod
    def get_candidate_spans(sent_lengths, max_sent_length, max_arg_width, mask=None):
        num_sentences = len(sent_lengths)
        # max_arg_width = max_sent_length  # max_arg_width = max_sent_length, since we don't need this constraint
        # Attention the order
        candidate_starts = torch.arange(0, max_sent_length).expand(num_sentences, max_arg_width, -1)
        candidate_width = torch.arange(0, max_arg_width).view(1, -1, 1)
        candidate_ends = candidate_starts + candidate_width

        candidate_starts = candidate_starts.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        candidate_ends = candidate_ends.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        actual_sent_lengths = sent_lengths.view(-1, 1).expand(-1, max_sent_length * max_arg_width)
        candidate_mask = candidate_ends < actual_sent_lengths.type(torch.LongTensor)
        if mask is not None:
            candidate_mask = candidate_mask * mask.type(candidate_mask.type())
        float_candidate_mask = candidate_mask.type(torch.LongTensor)

        candidate_starts = candidate_starts * float_candidate_mask
        candidate_ends = candidate_ends * float_candidate_mask
        return candidate_starts, candidate_ends, candidate_mask

    @staticmethod
    def exclusive_cumsum(input, exclusive=True):
        """
        :param input: input is the sentence lengths tensor.
        :param exclusive: exclude the last sentence length
        :return: the sum of y_i = x_1 + x_2 + ... + x_{i - 1} (i >= 1, and x_0 = 0)
        """
        assert exclusive is True
        if exclusive is True:
            exclusive_sent_lengths = torch.zeros(1).type(torch.LongTensor)
            result = torch.cumsum(torch.cat([exclusive_sent_lengths, input], 0)[:-1], 0).view(-1, 1)
        else:
            result = torch.cumsum(input, 0).view(-1, 1)
        return result

    @staticmethod
    def flatten_emb(emb):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        assert len(emb.size()) == 3
        flatted_emb = emb.contiguous().view(num_sentences * max_sentence_length, -1)
        return flatted_emb

    @staticmethod
    def flatten_emb_in_sentence(emb, batch_sentences_mask):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        flatted_emb = BiLSTMTaggerModel.flatten_emb(emb)
        return flatted_emb[batch_sentences_mask.view(num_sentences * max_sentence_length)]

    @staticmethod
    def get_span_emb(flatted_head_emb, flatted_context_emb, flatted_candidate_starts, flatted_candidate_ends,
                     config, dropout=0.0):
        batch_word_num = flatted_context_emb.size()[0]
        # span_num = flatted_candidate_starts.size()[0]  # candidate span num.
        # gather slices from embeddings according to indices
        span_start_emb = flatted_context_emb[flatted_candidate_starts]
        span_end_emb = flatted_context_emb[flatted_candidate_ends]

        span_sum_reps = span_start_emb + span_end_emb
        span_minus_reps = span_start_emb - span_end_emb
        span_emb_feature_list = [span_sum_reps, span_minus_reps]
        span_emb = torch.cat(span_emb_feature_list, 1)
        return span_emb

        span_emb_feature_list = [span_start_emb, span_end_emb]  # store the span vector representations for span rep.

        span_width = 1.0 + flatted_candidate_ends - flatted_candidate_starts  # [num_spans], generate the span width
        max_arg_width = config.max_arg_width
        # num_heads = config.num_attention_heads

        # get the span width feature emb
        span_width_index = span_width - 1
        span_width_emb = self.span_width_embedding(span_width_index.cuda())
        span_width_emb = torch.dropout(span_width_emb, dropout, self.training)
        span_emb_feature_list.append(span_width_emb)

        """head features"""
        cpu_flatted_candidte_starts = flatted_candidate_starts.cpu()
        # For all the i, where i in [begin, ..i, end] for span
        span_indices = torch.arange(0, max_arg_width).type(torch.LongTensor).view(1,
                                                                                  -1) + cpu_flatted_candidte_starts.view(
            -1, 1)
        # reset the position index to the batch_word_num index with index - 1
        span_indices = torch.clamp(span_indices, max=batch_word_num - 1)
        num_spans, spans_width = span_indices.size()[0], span_indices.size()[1]
        flatted_span_indices = span_indices.view(-1)  # so Huge!!!, column is the span?
        # if torch.cuda.is_available():
        flatted_span_indices = flatted_span_indices.cuda()
        span_text_emb = flatted_context_emb.index_select(0, flatted_span_indices).view(num_spans, spans_width, -1)
        span_indices_mask = self.sequence_mask(span_width, max_sent_length=max_arg_width)
        span_indices_mask = span_indices_mask.type(torch.Tensor).cuda()
        span_indices_log_mask = torch.log(span_indices_mask.type(torch.Tensor))
        # project context output to num head
        head_scores = self.context_projective_layer.forward(flatted_context_emb)
        # get span attention
        span_attention = head_scores.index_select(0, flatted_span_indices).view(num_spans, spans_width)
        expanded_span_indices_log_mask = span_indices_log_mask.cuda()
        span_attention = torch.add(span_attention, expanded_span_indices_log_mask).unsqueeze(2)  # control the span len
        span_attention = softmax(span_attention, dim=1)
        span_head_attention = torch.sum(span_attention * span_text_emb, 1)
        # span_text_emb = span_text_emb * span_indices_mask.unsqueeze(2). \
        #     expand(-1, -1, span_text_emb.size()[-1])
        # span_head_emb = torch.mean(span_text_emb, 1)
        span_emb_feature_list.append(span_head_attention)

        span_emb = torch.cat(span_emb_feature_list, 1)
        return span_emb, None, span_text_emb, span_indices, span_indices_mask

    def get_dse_unary_scores(self, span_emb, config, dropout, num_labels=1, name="span_scores"):
        input = span_emb
        for i, ffnn in enumerate(self.dse_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.dse_dropout_layers[i].forward(input)
        output = self.dse_unary_score_projection.forward(input)
        return output

    def get_arg_unary_scores(self, span_emb, config, dropout, num_labels=1, name="span_scores"):
        """
        Compute span score with FFNN(span embedding)
        :param span_emb: tensor of [num_sentences, num_spans, emb_size]
        :param config:
        :param dropout:
        :param num_labels:
        :param name:
        :return:
        """
        input = span_emb
        for i, ffnn in enumerate(self.arg_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.arg_dropout_layers[i].forward(input)
        output = self.arg_unary_score_projection.forward(input)
        return output

    def get_pred_unary_scores(self, span_emb, config, dropout, num_labels=1, name="pred_scores"):
        input = span_emb
        for i, ffnn in enumerate(self.pred_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.pred_dropout_layers[i].forward(input)
        output = self.pred_unary_score_projection.forward(input)
        return output

    def extract_spans(self, candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length, sort_spans,
                      enforce_non_crossing):
        """
        extract the topk span indices
        :param candidate_scores:
        :param candidate_starts:
        :param candidate_ends:
        :param topk: [num_sentences]
        :param max_sentence_length:
        :param sort_spans:
        :param enforce_non_crossing:
        :return: indices [num_sentences, max_num_predictions]
        """
        # num_sentences = candidate_scores.size()[0]
        # num_input_spans = candidate_scores.size()[1]
        max_num_output_spans = int(torch.max(topk))
        indices = [score.topk(k)[1] for score, k in zip(candidate_scores, topk)]
        output_span_indices_tensor = [F.pad(item, [0, max_num_output_spans - item.size()[0]], value=item[-1])
                                      for item in indices]
        output_span_indices_tensor = torch.stack(output_span_indices_tensor).cpu()
        return output_span_indices_tensor

    def batch_index_select(self, emb, indices):
        num_sentences = emb.size()[0]
        max_sent_length = emb.size()[1]

        flatten_emb = self.flatten_emb(emb)
        offset = (torch.arange(0, num_sentences) * max_sent_length).unsqueeze(1).cuda()
        return torch.index_select(flatten_emb, 0, (indices + offset).view(-1)) \
            .view(indices.size()[0], indices.size()[1], -1)

    def get_batch_topk(self, candidate_starts, candidate_ends, candidate_scores, topk_ratio, text_len,
                       max_sentence_length, sort_spans=False, enforce_non_crossing=True):
        num_sentences = candidate_starts.size()[0]
        max_sentence_length = candidate_starts.size()[1]

        topk = torch.floor(text_len.type(torch.FloatTensor) * topk_ratio)
        topk = torch.max(topk.type(torch.LongTensor), torch.ones(num_sentences).type(torch.LongTensor))

        # this part should be implemented with C++
        predicted_indices = self.extract_spans(candidate_scores, candidate_starts, candidate_ends, topk,
                                               max_sentence_length, sort_spans, enforce_non_crossing)
        predicted_starts = torch.gather(candidate_starts, 1, predicted_indices)
        predicted_ends = torch.gather(candidate_ends, 1, predicted_indices)
        predicted_scores = torch.gather(candidate_scores, 1, predicted_indices.cuda())
        return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices

    def get_dense_span_labels(self, dse_starts, dse_ends, span_starts, span_ends, span_labels, num_spans,
                              max_sentence_length):
        num_sentences = span_starts.size()[0]
        max_spans_num = span_starts.size()[1]

        span_starts = span_starts + 1 - self.sequence_mask(num_spans)
        sentence_indices = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).type(
            torch.LongTensor).cuda()

        sparse_indices = torch.cat([sentence_indices.unsqueeze(2),
                                    span_starts.unsqueeze(2), span_ends.unsqueeze(2),
                                    dse_starts.unsqueeze(2), dse_ends.unsqueeze(2)],
                                   dim=2)

        rank = 5
        dense_labels = torch.sparse.FloatTensor(
            sparse_indices.cpu().view(num_sentences * max_spans_num, rank).t(),
            span_labels.view(-1).type(torch.FloatTensor),
            torch.Size([num_sentences] + [max_sentence_length] * (rank - 1))
        ).to_dense()  # ok @kiro
        return dense_labels

    def gather_5d(self, params, indices):
        assert len(params.size()) == 5 and len(indices.size()) == 4
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c, indices_d, indices_e = indices.chunk(5, dim=3)
        result = params[indices_a, indices_b, indices_c, indices_d, indices_e]
        return result

    def gather_4d(self, params, indices):
        assert len(params.size()) == 4 and len(indices.size()) == 4
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c, indices_d = indices.chunk(4, dim=3)
        result = params[indices_a, indices_b, indices_c, indices_d]
        return result

    def get_srl_labels(self, dse_starts, dse_ends, arg_starts, arg_ends, labels, max_sentence_length):
        num_sentences = arg_starts.size()[0]
        max_arg_num = arg_starts.size()[1]
        max_pred_num = dse_starts.size(1)

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).unsqueeze(2).expand(-1, max_arg_num,
                                                                                              max_pred_num)
        expanded_arg_starts = arg_starts.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_arg_ends = arg_ends.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_dse_starts = dse_starts.unsqueeze(1).expand(-1, max_arg_num, -1)
        expanded_dse_ends = dse_ends.unsqueeze(1).expand(-1, max_arg_num, -1)
        # print(sentence_indices_2d.size(), expanded_arg_starts.size(), expanded_dse_starts.size())
        pred_indices = torch.cat([sentence_indices_2d.unsqueeze(3),
                                  expanded_arg_starts.unsqueeze(3), expanded_arg_ends.unsqueeze(3),
                                  expanded_dse_starts.unsqueeze(3), expanded_dse_ends.unsqueeze(3)],
                                 3)

        dense_srl_labels = self.get_dense_span_labels(
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
            max_sentence_length)  # ans

        srl_labels = self.gather_5d(dense_srl_labels, pred_indices.type(torch.LongTensor))  # TODO !!!!!!!!!!!!!
        # print(pred_indices, dense_srl_labels, srl_labels)
        # exit()
        return srl_labels

    def get_cons_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.cons_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            # if self.training:
            input = self.cons_dropout_layers[i].forward(input)
        output = self.cons_unary_score_projection.forward(input)
        num_spans = output.size(0)
        dummy_scores = torch.zeros([num_spans, 1]).cuda()
        cons_scores = torch.cat([dummy_scores, output], 1)
        return cons_scores

    def get_srl_unary_scores(self, span_emb, config, dropout, num_labels=1, name="span_scores"):
        input = span_emb
        for i, ffnn in enumerate(self.srl_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            # if self.training:
            input = self.srl_dropout_layers[i].forward(input)
        output = self.srl_unary_score_projection.forward(input)
        return output

    def eval_srl(self, srl_scores, gold_srl_index, srl_mask):
        """

        :param srl_scores: [num_sentences, num_args, num_preds, label_size - 1]  1 is the "O"
        :param gold_srl_index: [num_sentence, num_args, num_preds, 1] 1 is the index of srl label
        :return:
        """
        # print(srl_scores.size(), gold_srl_index.size(), srl_mask.size())
        srl_mask = srl_mask.type(torch.cuda.LongTensor)
        gold_srl_index = gold_srl_index.squeeze(-1).cuda()
        _, y_index = srl_scores.max(-1)
        y_index = y_index * srl_mask
        valid_sys_srl = y_index > 0
        sys_srl_num = int(valid_sys_srl.sum())
        valid_gold_srl = gold_srl_index * srl_mask > 0
        gold_srl_num = int(valid_gold_srl.sum())
        # print(y_index.size(), gold_srl_index.size(), valid_sys_srl.size())
        matched_srl_num = int(((y_index == gold_srl_index) * valid_sys_srl).sum())
        return matched_srl_num, sys_srl_num, gold_srl_num

    def get_srl_scores(self, arg_emb, pred_emb, num_labels, config, dropout):
        num_sentences = arg_emb.size()[0]
        num_args = arg_emb.size()[1]  # [batch_size, max_arg_num, arg_emb_size]
        num_preds = pred_emb.size()[1]  # [batch_size, max_pred_num, pred_emb_size]

        unsqueezed_arg_emb = arg_emb.unsqueeze(2)
        unsqueezed_pred_emb = pred_emb.unsqueeze(1)
        expanded_arg_emb = unsqueezed_arg_emb.expand(-1, -1, num_preds, -1)
        expanded_pred_emb = unsqueezed_pred_emb.expand(-1, num_args, -1, -1)
        pair_emb_list = [expanded_arg_emb, expanded_pred_emb]
        pair_emb = torch.cat(pair_emb_list, 3)  # concatenate the argument emb and pre emb
        pair_emb_size = pair_emb.size()[3]
        flat_pair_emb = pair_emb.view(num_sentences * num_args * num_preds, pair_emb_size)
        # get bilienar scorers
        # print(pred_emb.size(), arg_emb.size())
        # bilienar_scores = self.srl_bilinear_scorer.forward(pred_emb, arg_emb)
        # get unary scores
        flat_srl_scores = self.get_srl_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
                                                    "predicate_argument_scores")
        srl_scores = flat_srl_scores.view(num_sentences, num_args, num_preds, -1)
        # combine scores
        # softmax_weights = F.softmax(torch.cat([p for p in self.softmax_srl_scores_weights]), dim=0)
        # softmax_weights = torch.split(softmax_weights, 1)
        # srl_scores = softmax_weights[0] * bilienar_scores + softmax_weights[1] * srl_scores
        # unsqueezed_arg_scores, unsqueezed_pred_scores = \
        #     arg_scores.unsqueeze(2).unsqueeze(3), pred_scores.unsqueeze(1).unsqueeze(3)  # TODO ?
        # srl_scores = srl_scores + unsqueezed_arg_scores  # + unsqueezed_pred_scores
        dummy_scores = torch.zeros([num_sentences, num_args, num_preds, 1]).cuda()
        srl_scores = torch.cat([dummy_scores, srl_scores], 3)
        return srl_scores

    def get_predicates_according_to_index(self, pred_reps, candidate_pred_scores, candidate_pred_ids, pred_dense_index):
        """
        get the pred_emb, pred_score, pred_ids, the pred_dense_index can be pred or gold
        :param pred_reps: [num_sentence, max_sentence_length, pred_rep_dim]
        :param candidate_pred_scores: [num_sentence, max_sentence_length, 2]
        :param candidate_pred_ids: [num_sentence, max_sentence_length]
        :param pred_dense_index: [num_sentence, max_sentence_length]
        :return:
        """
        num_sentence = pred_dense_index.size(0)
        pred_nums = pred_dense_index.sum(dim=-1)
        max_pred_num = max(pred_nums)
        sparse_pred_index = pred_dense_index.nonzero()
        if max_pred_num == 0:  # if there is no predicate in this batch
            padded_sparse_pred_index = torch.zeros([num_sentence, 1]).type(torch.LongTensor)
            pred_nums[0] = 1  # give an artificial predicate
        else:
            padded_sparse_pred_index = torch.zeros([num_sentence, max_pred_num]).type(torch.LongTensor)
            # sent_wise_sparse_pred_index = sparse_pred_index.chunk(2, dim=-1)[1].view(-1)
            sent_wise_sparse_pred_index = sparse_pred_index[:, -1]
            offset = 0
            for i, pred_num in enumerate(pred_nums):
                pred_num = int(pred_num)
                ith_pred_index = sent_wise_sparse_pred_index[offset: offset + pred_num]
                padded_sparse_pred_index[i, :pred_num] = ith_pred_index
                offset += pred_num
        padded_sparse_pred_index = padded_sparse_pred_index.cuda()
        # get the returns
        pred_emb = self.batch_index_select(pred_reps, padded_sparse_pred_index)
        pred_ids = torch.gather(candidate_pred_ids, 1, padded_sparse_pred_index.cpu())
        pred_scores = self.batch_index_select(candidate_pred_scores, padded_sparse_pred_index)
        return pred_ids, pred_emb, pred_scores, pred_nums

    def get_gold_predicates(self, predicate_reps, candidate_pred_scores, candidate_pred_ids, gold_pred):
        gold_preds_num = gold_pred[2].sum(-1)
        # pred_emb = self.batch_index_select(predicate_reps, gold_predicate_indexes[2])
        sparse_pred_index = gold_pred[0]
        pred_emb = self.batch_index_select(predicate_reps, sparse_pred_index.cuda())
        predicates = torch.gather(candidate_pred_ids, 1, sparse_pred_index.cpu())
        pred_scores = self.batch_index_select(candidate_pred_scores, sparse_pred_index.cuda())[:, :, 1]
        return pred_emb, predicates, pred_scores, gold_preds_num

    def get_candidate_predicates_index(self, pred_scores, mask):
        y = softmax(pred_scores, -1)  # [batch_size, max_sentence_length, 2]
        max_y, max_indexes = y.max(dim=-1)  # max_indexes is the indices [batch_suze, max_sentence_length]
        max_indexes = max_indexes.type(torch.cuda.LongTensor) * mask.type(
            torch.cuda.LongTensor)  # since the padded position should be 0
        return max_indexes

    def eval_predicted_predicates(self, pred_predicates, gold_predicates, gold_pred_num=0):
        matched = pred_predicates * gold_predicates
        matched_num = int(matched.sum())
        total_gold_predicates = int(gold_predicates.sum())
        total_pred_predicates = int(pred_predicates.sum())
        assert total_gold_predicates == int(gold_pred_num.sum())
        return matched_num, total_pred_predicates, total_gold_predicates

    def get_pred_loss(self, pred_scores, gold_predicates, mask=None):
        assert len(pred_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        pred_scores = pred_scores.view(-1, pred_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = log_softmax(pred_scores, -1)
        # print(y)
        y_hat = gold_predicates.view(-1, 1)
        # print(y_hat)
        loss_flat = -torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        losses = loss_flat.view(*gold_predicates.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss

    def get_pred_focal_loss(self, pred_scores, gold_predicates, mask=None):
        assert len(pred_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        pred_scores = pred_scores.view(-1, pred_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = log_softmax(pred_scores, -1)
        # print(y)
        y_hat = gold_predicates.view(-1, 1)
        # print(y_hat)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * loss_flat
        losses = loss.view(*gold_predicates.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss

    def gather_3d(self, params, indices):
        assert len(params.size()) == 3 and len(indices.size()) == 3
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c = indices.chunk(3, dim=2)
        result = params[indices_a, indices_b, indices_c]
        return result.squeeze(2)

    def gather_2d(self, params, indices):
        assert len(params.size()) == 3 and len(indices.size()) == 2
        params = params.type(torch.LongTensor)
        indices_a, indices_b = indices.chunk(2, dim=1)
        # print(indices_a, indices_b)
        result = params[indices_a, indices_b, :]
        return result

    def sector_one_hot_2D(self, tensor):
        """
        :param tensor: [batch_size, max_sentence_length]
        :return: [batch_size, max_sentence_length, max_sentence_length]
        """
        batch_size, max_sentence_length = tensor.size(0), tensor.size(1)
        flag = self.config.max_arg_width > max_sentence_length
        results = []
        for t in tensor:
            tmp = []
            for i in range(0, self.config.max_arg_width):
                ith_m = t[i:]
                if flag is True:
                    if i <= max_sentence_length:
                        pass
                    else:
                        i = max_sentence_length
                ith_t = F.pad(ith_m, [0, 0, 0, i])
                tmp.append(ith_t)
            tmp = torch.stack(tmp, dim=0)
            # print(tmp.size())
            results.append(tmp)
        result = torch.stack(results, dim=0)
        # print(result.size())
        return result

    def ger_predicted_arg_boundary_by_prob_threshold(self, arg_boundary_scores, prob_threshold=0.5, mask=None):
        y = softmax(arg_boundary_scores, -1)  # [batch_size, max_sentence_length, 2]
        one_value = y[:, :, 1].squeeze(-1)
        max_y, max_index = y.max(dim=-1)
        prob_of_one_y = max_index * mask
        predicted_arg_boundary = (one_value > prob_threshold).type(max_index.type()) | prob_of_one_y.type(
            max_index.type())
        return max_index, predicted_arg_boundary.type(torch.IntTensor)

    def generate_span_scores(self, argu_scores, boundary_s_scores, boundary_e_scores, mask):
        # argu_scores: [batch_size, max_argu_number, 2]
        max_sentence_length = boundary_s_scores.size(1)
        boundary_s_scores = boundary_s_scores.unsqueeze(1).expand(-1, self.config.max_arg_width, -1,
                                                                  -1).contiguous().view(-1, 2)
        boundary_s_scores = boundary_s_scores[mask]
        boundary_e_scores = self.sector_one_hot_2D(boundary_e_scores).contiguous().view(-1, 2)
        boundary_e_scores = boundary_e_scores[mask]
        m_score = argu_scores + boundary_s_scores + boundary_e_scores
        return m_score

    def get_arg_boundary_fl_loss(self, arg_boundary_scores, gold_arg_boundary, mask=None):
        assert len(arg_boundary_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        arg_boundary_scores = arg_boundary_scores.view(-1, arg_boundary_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = log_softmax(arg_boundary_scores, -1)
        # print(y)
        y_hat = gold_arg_boundary.view(-1, 1)
        # print(y_hat)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * loss_flat
        losses = loss.view(*gold_arg_boundary.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss

    def eval_predicted_argument_boundaries(self, pred_arg_boundaries, gold_arg_boundarys, mask=None):
        pred_arg_boundaries = pred_arg_boundaries * mask.type(pred_arg_boundaries.type())
        matched = pred_arg_boundaries.cuda() * gold_arg_boundarys
        matched_num = int(matched.sum())
        total_gold_predicates = int(gold_arg_boundarys.sum())
        total_pred_predicates = int(pred_arg_boundaries.sum())
        return matched_num, total_pred_predicates, total_gold_predicates

    def sector_one_hot(self, tensor):
        """
        :param tensor: [batch_size, max_sentence_length]
        :return: [batch_size, max_sentence_length, max_sentence_length]
        """
        batch_size, max_sentence_length = tensor.size(0), tensor.size(1)
        flag = self.config.max_arg_width > max_sentence_length
        results = []
        for t in tensor:
            tmp = []
            for i in range(0, self.config.max_arg_width):
                ith_m = t[i:]
                if flag is True:
                    if i <= max_sentence_length:
                        pass
                    else:
                        i = max_sentence_length
                ith_t = F.pad(ith_m, [0, i])
                tmp.append(ith_t)
            tmp = torch.stack(tmp, dim=0)
            # print(tmp.size())
            results.append(tmp)
        result = torch.stack(results, dim=0)
        # print(result.size())
        return result

    def get_arg_boundary_by_threshold(self, boundary_s_scores, boundary_e_scores, threshold=0.0):
        # argu_scores: [batch_size, max_argu_number, 2]
        batch_size, max_sentence_length = boundary_s_scores.size(0), boundary_s_scores.size(1)
        sp = softmax(boundary_s_scores, -1)
        ep = softmax(boundary_e_scores, -1)
        if self.config.pruning_by_arg_prob:
            ps, pe = sp[:, :, 1].squeeze(-1), ep[:, :, 1].squeeze(-1)
            ps_m = ps.unsqueeze(1).expand(-1, self.config.max_arg_width, -1).contiguous()
            ps_e = self.sector_one_hot(pe)
            multi_m = ps_m * ps_e
            indexes = (multi_m >= threshold).type(torch.LongTensor).view(batch_size,
                                                                         self.config.max_arg_width * max_sentence_length)
        else:
            s_value, s_index = sp.max(dim=-1)
            e_value, e_index = ep.max(dim=-1)
            so = s_index.unsqueeze(1).expand(-1, self.config.max_arg_width, -1).contiguous()
            eo = self.sector_one_hot(e_index)
            indexes = (so * eo).type(torch.LongTensor).view(batch_size, self.config.max_arg_width * max_sentence_length)
        return indexes

    def get_candidate_argument_index(self, argu_scores, mask):
        # argu_scores: [batch_size, max_argu_number, 2]
        y = softmax(argu_scores, -1)
        y_value, max_indexes = y.max(dim=-1)  #
        max_indexes = max_indexes.type(torch.cuda.LongTensor) * mask
        return max_indexes

    def get_candidate_argument_index_by_threshold(self, argu_scores, boundary_s_scores, boundary_e_scores,
                                                  mask, threshold=0.0):
        # argu_scores: [batch_size, max_argu_number, 2]
        max_sentence_length = boundary_s_scores.size(1)
        yp = softmax(argu_scores, -1)[:, :, 1].squeeze(-1)
        sp = softmax(boundary_s_scores, -1)[:, :, 1].squeeze(-1)
        ep = softmax(boundary_e_scores, -1)[:, :, 1].squeeze(-1)
        mp = yp * sp.unsqueeze(1).expand(-1, self.config.max_arg_width, -1).reshape(yp.size()) * \
             self.sector_one_hot(ep).reshape(yp.size())
        max_indexes = (mp > threshold).type(torch.cuda.LongTensor) * mask
        return max_indexes

    def get_gold_cons_index(self, gold_labels, max_sentence_length, candidate_argu_mask):
        span_starts, span_ends, num_spans, span_labels = gold_labels[0], gold_labels[1], gold_labels[2], gold_labels[3]
        # print(gold_labels)
        num_sentences = span_starts.size(0)
        max_spans_num = span_starts.size(1)
        x_max_spans_num = candidate_argu_mask.size(1)

        gold_argu_num = int(num_spans.sum())
        flat_indices = (span_ends - span_starts) * max_sentence_length + span_starts

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).cuda()
        sparse_argument_indices = torch.cat([sentence_indices_2d.unsqueeze(2), flat_indices.unsqueeze(2)], 2)

        sparse_argument_indices = torch.cat([sparse_argument_indices[i, :num_spans[i], :] for i in range(num_sentences)]
                                            , dim=0)
        span_labels = span_labels.cpu()
        span_labels = torch.cat([span_labels[i, :num_spans[i]] for i in range(num_sentences)], dim=0)
        # print(sparse_argument_indices.size(), gold_argu_num, span_labels.size(), num_sentences, x_max_spans_num)
        dense_gold_argus = torch.sparse.FloatTensor(
            sparse_argument_indices.cpu().view(gold_argu_num, -1).t(),
            span_labels,
            torch.Size([num_sentences, x_max_spans_num])
        ).to_dense()
        # print(gold_argu_num, int((dense_gold_argus > 0).sum()), (span_labels > 0).sum())
        assert gold_argu_num == int((dense_gold_argus > 0).sum())
        return dense_gold_argus.cuda()

    def get_gold_dense_argu_index(self, gold_labels, max_sentence_length, candidate_argu_mask):
        span_starts, span_ends, num_spans = gold_labels[0], gold_labels[1], gold_labels[2]
        print('during generating gold dense argu index span-starts\n', span_starts)
        print('during generating gold dense argu index span-starts\n', span_ends)
        print('during generating gold dense argu index num-span\n', num_spans)
        num_sentences = span_starts.size(0)
        max_spans_num = span_starts.size(1)
        # x_num_sentences = candidate_argu_mask.size(0)
        x_max_spans_num = candidate_argu_mask.size(1)

        gold_argu_num = int(num_spans.sum())

        # total_span_num = num_sentences * max_spans_num
        flat_indices = (span_ends - span_starts) * max_sentence_length + span_starts
        print('flat_indices', flat_indices)

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).cuda()
        sparse_argument_indices = torch.cat([sentence_indices_2d.unsqueeze(2), flat_indices.unsqueeze(2)], 2)

        sparse_argument_indices = torch.cat(
            [sparse_argument_indices[i, :num_spans[i], :] for i in range(num_sentences)], dim=0)
        # print(sparse_argument_indices.size(), sparse_argument_indices)
        # print(span_starts, span_ends, sparse_argument_indices)
        # print(sparse_argument_indices.size(), gold_argu_num, x_max_spans_num)
        dense_gold_argus = torch.sparse.FloatTensor(sparse_argument_indices.cpu().view(gold_argu_num, -1).t(),
                                                    torch.LongTensor([1] * gold_argu_num).view(-1),
                                                    torch.Size([num_sentences, x_max_spans_num])).to_dense()
        # dense_gold_argus = dense_gold_argus * candidate_argu_mask  # TODO
        # print(num_spans, gold_argu_num, int(dense_gold_argus.sum()))
        assert gold_argu_num == int(dense_gold_argus.sum())
        # print(dense_gold_argus.nonzero(), candidate_argu_mask.nonzero())
        # exit()
        dense_gold_argus = (dense_gold_argus > 0).type(torch.cuda.LongTensor)
        # print(dense_gold_argus.size(), dense_gold_argus)
        # exit()
        return dense_gold_argus

    def eval_predicted_cons(self, sys_cons, gold_cons, mask):
        sys_cons = sys_cons * mask.type(sys_cons.type())
        matched_num = int((sys_cons == gold_cons).sum())
        pred_num = int((sys_cons > 0).sum())
        gold_num = int((gold_cons > 0).sum())
        return matched_num, pred_num, gold_num

    def eval_predicted_arguments(self, sys_arguments, gold_arguments, mask):
        sys_arguments = sys_arguments * mask.type(sys_arguments.type())
        matched = sys_arguments * gold_arguments
        matched_num = int(matched.sum())
        total_pred_arguments = int(sys_arguments.sum())
        total_gold_arguments = int(gold_arguments.sum())
        return matched_num, total_pred_arguments, total_gold_arguments

    def get_gold_arguments_index(self, gold_labels, max_sentence_length):
        span_starts, span_ends = gold_labels[1], gold_labels[2]
        num_sentences = span_starts.size(0)
        max_spans_num = span_starts.size(1)

        total_span_num = num_sentences * max_spans_num
        sparse_indices = self.generate_span_indices(span_starts, span_ends)
        # print("gold labels", gold_labels)
        # print("gold span indexes", sparse_indices)
        rank = 3
        dense_spans = torch.sparse.FloatTensor(
            sparse_indices.cpu().view(num_sentences * max_spans_num, rank).t(),
            torch.LongTensor([1] * total_span_num).view(-1),
            torch.Size([num_sentences] + [max_sentence_length] * (rank - 1))) \
            .to_dense()  # ok @kiro
        # print("gold span values", num_sentences, max_spans_num, max_sentence_length, torch.LongTensor([1] * total_span_num).view(-1))
        # print("generate gold spans", dense_spans.size(), dense_spans)
        dense_spans = dense_spans > 0
        return dense_spans.type(torch.LongTensor)

    def get_arguments_according_to_no_one_hot_index(
            self, argument_reps, argument_scores,
            candidate_span_starts, candidate_span_ends, candidate_span_ids,
            argu_dense_index):
        """
        return predicted arguments
        :param argument_reps: flatted; [batch_total_span_num]
        :param argument_scores:
        :param candidate_span_starts:
        :param candidate_span_ends:
        :param candidate_span_ids:
        :param argu_dense_index:
        :return:
        """
        num_sentences = argu_dense_index.size(0)
        num_args = (argu_dense_index > 0).sum(-1)
        sparse_argu_index = argu_dense_index.nonzero()
        max_argu_num = int(max(num_args))
        if max_argu_num == 0:
            padded_sparse_argu_index = torch.zeros([num_sentences, 1]).type(torch.LongTensor)
            num_args[0] = 1
        else:
            padded_sparse_argu_index = torch.zeros([num_sentences, max_argu_num]).type(torch.LongTensor)
            sent_wise_sparse_argu_index = sparse_argu_index[:, -1]
            offset = 0
            for i, argu_num in enumerate(num_args):
                argu_num = int(argu_num)
                ith_argu_index = sent_wise_sparse_argu_index[offset: offset + argu_num]
                padded_sparse_argu_index[i, :argu_num] = ith_argu_index
                offset += argu_num
        padded_sparse_argu_index = padded_sparse_argu_index.cuda()

        argu_starts = torch.gather(candidate_span_starts, 1, padded_sparse_argu_index.cpu())
        argu_ends = torch.gather(candidate_span_ends, 1, padded_sparse_argu_index.cpu())
        argu_scores = self.batch_index_select(argument_scores, padded_sparse_argu_index)
        arg_span_indices = torch.gather(candidate_span_ids, 1,
                                        padded_sparse_argu_index)  # [num_sentences, max_num_args]
        arg_emb = argument_reps.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1
        )
        return arg_emb, argu_scores, argu_starts, argu_ends, num_args

    def get_arguments_according_to_index(self, argument_reps, argument_scores,
                                         candidate_span_starts, candidate_span_ends, candidate_span_ids,
                                         argu_dense_index):
        """
        return predicted arguments
        :param argument_reps: flatted; [batch_total_span_num]
        :param argument_scores:
        :param candidate_span_starts:
        :param candidate_span_ends:
        :param candidate_span_ids:
        :param argu_dense_index:
        :return:
        """
        num_sentences = argu_dense_index.size(0)
        num_args = argu_dense_index.sum(-1)
        sparse_argu_index = argu_dense_index.nonzero()
        max_argu_num = int(max(num_args))
        if max_argu_num == 0:
            padded_sparse_argu_index = torch.zeros([num_sentences, 1]).type(torch.LongTensor)
            num_args[0] = 1
        else:
            padded_sparse_argu_index = torch.zeros([num_sentences, max_argu_num]).type(torch.LongTensor)
            sent_wise_sparse_argu_index = sparse_argu_index[:, -1]
            offset = 0
            for i, argu_num in enumerate(num_args):
                argu_num = int(argu_num)
                ith_argu_index = sent_wise_sparse_argu_index[offset: offset + argu_num]
                padded_sparse_argu_index[i, :argu_num] = ith_argu_index
                offset += argu_num
        padded_sparse_argu_index = padded_sparse_argu_index.cuda()

        argu_starts = torch.gather(candidate_span_starts, 1, padded_sparse_argu_index.cpu())
        argu_ends = torch.gather(candidate_span_ends, 1, padded_sparse_argu_index.cpu())
        # print(torch.cat([predicted_starts.unsqueeze(2), predicted_ends.unsqueeze(2)], dim=2))
        # print(span_ends.size(), argument_scores.size())
        # argu_scores = torch.gather(argument_scores, 1, padded_sparse_argu_index)
        # print("xxx", argument_scores.size(), padded_predicted_predicates_index.size(), argument_scores, padded_predicted_predicates_index)
        argu_scores = self.batch_index_select(argument_scores, padded_sparse_argu_index)
        # argu_scores = torch.gather(argument_scores, 1,
        #                                 padded_sparse_argu_index.unsqueeze(2).expand(-1, -1, 2).cuda())
        arg_span_indices = torch.gather(candidate_span_ids, 1,
                                        padded_sparse_argu_index)  # [num_sentences, max_num_args]
        arg_emb = argument_reps.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1
        )
        return arg_emb, argu_scores, argu_starts, argu_ends, num_args

    def generate_span_indices(self, starts, ends):
        # starts, ends, [num_sentences, max_argument_num]
        num_sentences = starts.size()[0]
        max_arg_num = starts.size()[1]

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_arg_num).cuda()
        pred_indices = torch.cat([sentence_indices_2d.unsqueeze(2), starts.unsqueeze(2),
                                  ends.unsqueeze(2)], 2)
        return pred_indices

    def generate_flat_span_indices(self, ids):
        num_sentences = ids.size(0)
        max_arg_num = ids.size(1)
        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_arg_num).cuda()
        # print(sentence_indices_2d.size(), ids.size())
        flat_indices = torch.cat([sentence_indices_2d, ids], dim=-1)
        return flat_indices

    def get_argument_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num, 1]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        loss_flat = -torch.gather(y, dim=-1, index=y_hat)
        losses = loss_flat.view(*gold_argument_index.size())
        losses = losses * candidate_argu_mask.float()
        loss = losses.sum()
        return loss

    def get_argument_focal_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num, 1]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        if self.training:
            ## randomly choose negative samples
            randn_neg_sample = torch.randint(0, 100, gold_argument_index.size()).type(candidate_argu_mask.type())
            randn_neg_sample = randn_neg_sample > int(self.config.neg_threshold)  # randomly
            randn_neg_sample = randn_neg_sample | gold_argument_index.type(randn_neg_sample.type())
            candidate_argu_mask = candidate_argu_mask * randn_neg_sample.type(candidate_argu_mask.type())
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * loss_flat
        losses = loss.view(*gold_argument_index.size())
        losses = losses * candidate_argu_mask.float()
        loss = losses.sum()
        return loss

    def get_candidate_arguments(self, argument_reps, candidate_span_ids, span_indexes):
        arg_span_indices = torch.gather(candidate_span_ids, 1, span_indexes.cuda())  # [num_sentences, max_num_args]
        arg_emb = argument_reps.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1)  # [num_sentences, max_num_args, emb]
        return arg_emb

    def get_srl_softmax_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        # print(srl_scores.size(), srl_labels.size(), num_predicted_args, num_predicted_preds)
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        srl_mask = (args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1)

        srl_scores = srl_scores.view(-1, num_labels)
        srl_labels = (srl_labels.view(-1, 1)).cuda()
        output = log_softmax(srl_scores, 1)

        negative_log_likelihood_flat = -torch.gather(output, dim=1, index=srl_labels).view(-1)
        srl_loss_mask = (srl_mask.view(-1) == 1).nonzero()
        # print(srl_loss_mask, srl_labels, negative_log_likelihood_flat)
        # print("srl loss", negative_log_likelihood_flat.size(), srl_loss_mask.size())
        # print(type(srl_mask), srl_mask)
        if int(srl_labels.sum()) == 0 or int(sum(srl_loss_mask)) == 0:
            # print("xxx")
            # print(negative_log_likelihood_flat)
            loss = negative_log_likelihood_flat.sum()
            # print(loss)
            # print("xxx done")
            return loss, srl_mask
        # srl_loss_mask = srl_loss_mask.view(-1)
        # print(negative_log_likelihood_flat.size(), srl_loss_mask.size(), srl_loss_mask)
        # negative_log_likelihood = torch.gather(negative_log_likelihood_flat, dim=0, index=srl_loss_mask)
        srl_mask = srl_mask.type(torch.cuda.FloatTensor)
        negative_log_likelihood_flat = negative_log_likelihood_flat.view(srl_mask.size()) * srl_mask
        loss = negative_log_likelihood_flat.sum()
        return loss, srl_mask

    def get_srl_softmax_focal_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        # print(srl_scores.size(), srl_labels.size(), num_predicted_args, num_predicted_preds)
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        srl_mask = (args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1)

        srl_scores = srl_scores.view(-1, num_labels)
        srl_labels = (srl_labels.view(-1, 1)).cuda()
        output = log_softmax(srl_scores, 1)
        # print(srl_labels.size(), srl_labels)
        # print("check labels", (srl_labels > 2).sum())
        negative_log_likelihood_flat = torch.gather(output, dim=1, index=srl_labels).view(-1)

        pt = negative_log_likelihood_flat.exp()  # pt is the softmax score
        negative_log_likelihood_flat = \
            -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * negative_log_likelihood_flat

        # srl_loss_mask = (srl_mask.view(-1) == 1).nonzero()
        # if int(srl_labels.sum()) == 0 or int(sum(srl_loss_mask)) == 0:
        #     loss = negative_log_likelihood_flat.sum()
        #     return loss, srl_mask
        srl_mask = srl_mask.type(torch.cuda.FloatTensor)
        negative_log_likelihood_flat = negative_log_likelihood_flat.view(srl_mask.size()) * srl_mask
        loss = negative_log_likelihood_flat.sum()
        return loss, srl_mask

    def get_sys_cons(self, lstm_out, masks, num_sentences, sent_lengths, max_sent_length, arguments_index=None):
        # compute the arg scores of the cons starts and ends
        arg_starts_scores, arg_ends_scores, argu_reps = self.cons_classifier.forward(lstm_out)
        """Get the candidate arguments"""
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width, None)

        predict_dict = dict()

        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_context_output = BiLSTMTaggerModel.flatten_emb_in_sentence(
            argu_reps, byte_sentence_mask)  # cuda type
        # batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()

        candidate_span_emb = BiLSTMTaggerModel.get_span_emb(flatted_context_output, flatted_context_output,
                                                            flatted_candidate_starts, flatted_candidate_ends,
                                                            self.config, dropout=self.dropout)
        candidate_span_emb = self.cons_reps_drop(torch.relu(self.cons_reps.forward(candidate_span_emb)))
        # Get the span ids
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number)
        candidate_span_ids = \
            torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                     torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        predict_dict.update({"candidate_cons_starts": candidate_starts, "candidate_cons_ends": candidate_ends})
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_arg_scores = self.get_cons_scores(candidate_span_emb)
        # if self.config.joint:
        #     flatted_candidate_arg_scores = self.generate_span_scores(
        #         flatted_candidate_arg_scores, arg_starts_scores, arg_ends_scores,
        #         flatted_candidate_mask
        #     )
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_mask = candidate_mask.type(torch.Tensor).unsqueeze(2).expand(-1, -1, candidate_arg_scores.size(-1)).cuda()
        candidate_arg_scores = candidate_arg_scores * spans_mask
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted arguments
        predicted_arguments_index = \
            self.get_candidate_argument_index(candidate_arg_scores,
                                              candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 4. predicte the cons
        arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
            self.get_arguments_according_to_no_one_hot_index(
                candidate_span_emb, candidate_arg_scores,
                candidate_starts, candidate_ends, candidate_span_ids,
                predicted_arguments_index
            )
        predict_dict.update({
            "sys_cons_scores": arg_scores,
            "sys_cons_starts": arg_starts,
            "sys_cons_ends": arg_ends,
            "sys_cons_nums": num_args
        })
        if arguments_index is not None:
            # 4. predicte the cons
            arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
                self.get_arguments_according_to_no_one_hot_index(
                    candidate_span_emb, candidate_arg_scores,
                    candidate_starts, candidate_ends, candidate_span_ids,
                    arguments_index
                )
            predict_dict.update({
                "sys_argu_scores": arg_scores,
                "sys_argu_starts": arg_starts,
                "sys_argu_ends": arg_ends,
                "sys_argu_nums": num_args
            })
        return predict_dict, predicted_arguments_index, flatted_candidate_arg_scores

    def cons_forward(self, cons_data):
        input_data, cons_spans, cons_boundary = cons_data
        sent_lengths, words, chars = input_data
        cons_starts, cons_ends, cons_labels, cons_num = cons_spans
        cons_starts_one_hot, cons_ends_one_hot = cons_boundary

        num_sentences, max_sent_length = words.size()[0], words.size()[1]
        word_embeddings, char_embeddings = self.word_embedding(words), self.get_char_cnn_embeddings(chars)

        context_embeddings = torch.cat([word_embeddings, char_embeddings], dim=2)
        context_embeddings = torch.dropout(context_embeddings, self.lexical_dropout, self.training)

        masks = self.init_masks(num_sentences, sent_lengths)
        lstm_out, _ = self.cons_bilstm(context_embeddings, masks)

        predict_dict = dict()

        # compute the arg scores of the cons starts and ends
        arg_starts_scores, arg_ends_scores, argu_reps = self.cons_classifier.forward(lstm_out)
        # 2. get the arg starts and ends with probability threshold
        true_pred_starts, predicted_arg_starts_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_starts_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        true_pred_ends, predicted_arg_ends_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_ends_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        # 3. compute the losses of the candidate argument starts and ends
        arg_starts_loss = self.get_arg_boundary_fl_loss(arg_starts_scores, cons_starts_one_hot, mask=masks)
        arg_ends_loss = self.get_arg_boundary_fl_loss(arg_ends_scores, cons_ends_one_hot, mask=masks)
        # 4. evaluate the predicted argument boundary
        num_match_arg_starts, num_pred_arg_starts, num_gold_arg_starts = \
            self.eval_predicted_argument_boundaries(true_pred_starts, cons_starts_one_hot, mask=masks)
        num_match_arg_ends, num_pred_arg_ends, num_gold_arg_ends = \
            self.eval_predicted_argument_boundaries(true_pred_ends, cons_ends_one_hot, mask=masks)
        predict_dict.update({"matched_cons_starts": num_match_arg_starts,
                             "pred_cons_starts": num_pred_arg_starts,
                             "gold_cons_starts": num_gold_arg_starts,

                             "matched_cons_ends": num_match_arg_ends,
                             "pred_cons_ends": num_pred_arg_ends,
                             "gold_cons_ends": num_gold_arg_ends}
                            )
        arg_boundary_mask = None
        # print(self.config.pruning_by_arg_prob)
        if self.config.pruning_by_arg_prob:
            arg_boundary_mask = self.get_arg_boundary_by_threshold(
                arg_starts_scores, arg_ends_scores,
                self.config.arg_boundary_prob_threshold
            )
        """Get the candidate arguments"""
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width, arg_boundary_mask)

        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_context_output = BiLSTMTaggerModel.flatten_emb_in_sentence(
            argu_reps, byte_sentence_mask)  # cuda type
        # batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()

        candidate_span_emb = BiLSTMTaggerModel.get_span_emb(flatted_context_output, flatted_context_output,
                                                            flatted_candidate_starts, flatted_candidate_ends,
                                                            self.config, dropout=self.dropout)
        candidate_span_emb = self.cons_reps_drop(torch.relu(self.cons_reps.forward(candidate_span_emb)))
        # Get the span ids
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number)
        candidate_span_ids = \
            torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                     torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        predict_dict.update({"candidate_cons_starts": candidate_starts, "candidate_cons_ends": candidate_ends})
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_arg_scores = self.get_cons_scores(candidate_span_emb)
        # if self.config.joint:
        #     flatted_candidate_arg_scores = self.generate_span_scores(
        #         flatted_candidate_arg_scores, arg_starts_scores, arg_ends_scores,
        #         flatted_candidate_mask
        #     )
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_mask = candidate_mask.type(torch.Tensor).unsqueeze(2).expand(-1, -1, candidate_arg_scores.size(-1)).cuda()
        candidate_arg_scores = candidate_arg_scores * spans_mask
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted arguments
        predicted_arguments_index = \
            self.get_candidate_argument_index(candidate_arg_scores,
                                              candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 2. eval predicted argument
        dense_gold_argus_index = \
            self.get_gold_cons_index((cons_starts, cons_ends, cons_num, cons_labels), max_sent_length,
                                     candidate_mask.type(torch.LongTensor).view(num_sentences, -1))
        matched_argu_num, sys_argu_num, gold_argu_num = \
            self.eval_predicted_cons(predicted_arguments_index, dense_gold_argus_index, mask=candidate_mask)
        predict_dict.update({"matched_cons_num": matched_argu_num,
                             "sys_cons_num": sys_argu_num,
                             "gold_cons_num": gold_argu_num})
        # 3. compute argument loss
        argument_loss = self.get_argument_focal_loss(
            candidate_arg_scores, dense_gold_argus_index,
            candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        return argument_loss

    def batch_gcn_inputs(self, cons_trees, num_sentences):
        cons_nodes = [tree.node_idx for tree in cons_trees]
        cons_node_chars = [tree.node_char_idx for tree in cons_trees]
        cons_heads = [tree.heads for tree in cons_trees]
        cons_indicator = [tree.indicator for tree in cons_trees]
        word_position = [tree.word_position for tree in cons_trees]
        # padding cons labels idx and heads
        sent_lengths = [len(x) for x in cons_nodes]
        max_cons_nodes_length = max(sent_lengths)
        max_node_char_length = max([x.shape[1] for x in cons_node_chars])
        padded_cons_nodes = np.zeros([num_sentences, max_cons_nodes_length], dtype=np.int)
        padded_node_chars = np.zeros([num_sentences, max_cons_nodes_length, max_node_char_length], dtype=np.int)
        padded_words_indicator = []
        padded_words_position = []
        assert num_sentences == len(cons_nodes)
        for i in range(num_sentences):
            # print(i, word_position[i], cons_indicator[i], cons_nodes[i], '\n')
            padded_cons_nodes[i][:sent_lengths[i]] = cons_nodes[i]
            padded_node_chars[i][:cons_node_chars[i].shape[0], :cons_node_chars[i].shape[1]] = cons_node_chars[i]
            padded_words_indicator.append(torch.from_numpy(np.array(cons_indicator[i])).type(torch.bool))
            padded_words_position.append(torch.from_numpy(np.array(word_position[i], dtype=np.int)))

        sent_lengths = torch.from_numpy(np.array(sent_lengths)).view(-1)
        masks = self.init_masks(num_sentences, sent_lengths)
        # l = (masks.data.cpu().numpy() == 1).astype(np.int).sum(1)
        adj = inputs_to_tree_reps(cons_heads, None, max_cons_nodes_length, sent_lengths, -1, pheads=None)
        # generate the input of GCN module
        node_index = torch.from_numpy(np.array(padded_cons_nodes).astype(np.int))
        node_char_index = torch.from_numpy(np.array(padded_node_chars)).cuda()

        # TODO, it is very important to add dropout over the input layer
        gcn_input = torch.dropout(
            torch.cat([self.word_embedding(node_index.cuda()), self.get_char_cnn_embeddings(node_char_index)], dim=-1),
            self.lexical_dropout, self.training
        )
        word_positions = pad_sequence(padded_words_position, batch_first=True)
        word_mask = pad_sequence(padded_words_indicator, batch_first=True)
        words_numbers = word_mask.sum(1)
        word_index = torch.split(node_index[word_mask], words_numbers.tolist())
        word_index = pad_sequence(word_index, batch_first=True)
        # print(word_index.size())
        return adj, gcn_input, masks, (word_mask, words_numbers, word_index)

    def dep_forward(self, sent_lengths, input_seq, gold_dse, gold_arg, gold_orl, cons_trees, dep):
        words, chars = input_seq
        num_sentences, max_sent_length = words.size()[0], words.size()[1]

        word_embeddings, char_embeddings = self.word_embedding(words), self.get_char_cnn_embeddings(chars)

        context_embeddings = torch.cat([word_embeddings, char_embeddings], dim=2)
        context_embeddings = torch.dropout(context_embeddings, self.lexical_dropout, self.training)

        dep_loss, _ = self.implicit_dependency_representation(
            num_sentences, context_embeddings, sent_lengths, dep)
        return None, dep_loss

    def forward(self, sent_lengths, input_seq, gold_dse, gold_arg, gold_orl, cons_trees, dep=None, hot_start=False):
        seqs_text, words, chars = input_seq
        num_sentences, max_sent_length = words.size()[0], words.size()[1]

        word_embeddings, char_embeddings = self.word_embedding(words), self.get_char_cnn_embeddings(chars)

        context_embeddings = torch.cat([word_embeddings, char_embeddings], dim=2)
        if self.config.use_bert:
            bert_embeddings = self.bert_input.forward(seqs_text)
            context_embeddings = torch.cat([context_embeddings, bert_embeddings], dim=2)
        context_embeddings = torch.dropout(context_embeddings, self.lexical_dropout, self.training)
        orl_input = context_embeddings
        masks = self.init_masks(num_sentences, sent_lengths)

        if self.config.mtl_cons:
            """Implicit Features"""
            cons_lstm_out, features = self.cons_bilstm(context_embeddings, masks)

            normed_weights = F.softmax(torch.cat([param for param in self.isr_weights]), dim=0)
            normed_weights = torch.split(normed_weights, 1)  # split_size_or_sections=1, split_size=1)  # 0.3.0
            isr = self.isr_gamma * \
                  sum([normed_weights[i] * features[i] for i in range(self.cons_num_lstm_layers)])
            orl_input = torch.cat([orl_input,
                                   torch.dropout(isr, self.config.feature_dropout, self.training)], dim=-1)
        if self.config.mtl_dep:
            dep_representations, arc_logits = self.implicit_dependency_representation.get_reps(context_embeddings, masks)
            orl_input = torch.cat([orl_input,
                                   torch.dropout(dep_representations, self.config.feature_dropout, self.training)],
                                  dim=-1)
        if self.config.use_cons_gcn:
            adj, gcn_input, gcn_masks, words_position = self.batch_gcn_inputs(cons_trees, num_sentences)
            gcn_output = self.cons_gcn.forward(adj, gcn_input, gcn_masks)

            words_mask, words_numbers, word_index = words_position
            # print(word_index.type(), words.cpu().type())
            # print(words, word_index)
            # assert word_index.equal(words.cpu())
            gcn_outputs = torch.split(gcn_output[words_mask], words_numbers.tolist())
            gcn_outputs = pad_sequence(gcn_outputs, batch_first=True)
            orl_input = torch.cat([orl_input,
                                   torch.dropout(gcn_outputs, self.config.auto_feature_dropout, self.training)], dim=-1)
        if self.config.use_dep_gcn:
            assert self.config.mtl_dep
            adj = F.softmax(arc_logits, dim=-1)
            # heads = [item.heads[1:] for item in dep]
            # adj = inputs_to_tree_reps(heads, None, max_sent_length, sent_lengths, -1, pheads=None)
            dep_gcn_output = self.dep_gcn.forward(adj, context_embeddings, masks)
            orl_input = torch.cat([orl_input,
                                   torch.dropout(dep_gcn_output, self.config.auto_feature_dropout, self.training)], dim=-1)

        """Implicit Features End"""
        lstm_out, _ = self.bilstm(orl_input, masks)

        predict_dict = dict()

        # DSE starts and ends prediction
        gold_dse_starts, gold_dse_ends, gold_dse_num, gold_dse_starts_one_hot, gold_dse_ends_one_hot = gold_dse

        dse_starts_scores = self.dse_start_scorer.forward(lstm_out)
        dse_ends_scores = self.dse_end_scorer.forward(lstm_out)
        # 2. get the dse starts and ends with probability threshold
        true_pred_dse_starts, predicted_dse_starts_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            dse_starts_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        true_pred_dse_ends, predicted_dse_ends_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            dse_ends_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        # 3. compute the losses of the candidate dse starts and ends
        dse_starts_loss = self.get_arg_boundary_fl_loss(dse_starts_scores, gold_dse_starts_one_hot, mask=masks)
        dse_ends_loss = self.get_arg_boundary_fl_loss(dse_ends_scores, gold_dse_ends_one_hot, mask=masks)
        # 4. evaluate the predicted dse boundary
        num_match_dse_starts, num_pred_dse_starts, num_gold_dse_starts = \
            self.eval_predicted_argument_boundaries(true_pred_dse_starts, gold_dse_starts_one_hot, mask=masks)
        num_match_dse_ends, num_pred_dse_ends, num_gold_dse_ends = \
            self.eval_predicted_argument_boundaries(true_pred_dse_ends, gold_dse_ends_one_hot, mask=masks)
        predict_dict.update({"matched_dse_starts": num_match_dse_starts,
                             "pred_dse_starts": num_pred_dse_starts,
                             "gold_dse_starts": num_gold_dse_starts,

                             "matched_dse_ends": num_match_dse_ends,
                             "pred_dse_ends": num_pred_dse_ends,
                             "gold_dse_ends": num_gold_dse_ends}
                            )

        dse_reps = self.dse_reps_drop_0(torch.relu(self.dse_reps_0.forward(lstm_out)))
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_dse_starts, candidate_dse_ends, candidate_dse_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width)

        flatted_candidate_dse_mask = candidate_dse_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_dse_starts = candidate_dse_starts + batch_word_offset
        flatted_candidate_dse_starts = flatted_candidate_dse_starts.view(-1)[flatted_candidate_dse_mask].type(
            torch.LongTensor)
        flatted_candidate_dse_ends = candidate_dse_ends + batch_word_offset
        flatted_candidate_dse_ends = flatted_candidate_dse_ends.view(-1)[flatted_candidate_dse_mask].type(
            torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_dse_context_output = BiLSTMTaggerModel.flatten_emb_in_sentence(
            dse_reps, byte_sentence_mask)  # cuda type
        # batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_dse_starts, flatted_candidate_dse_ends = \
                flatted_candidate_dse_starts.cuda(), flatted_candidate_dse_ends.cuda()

        candidate_dse_span_emb = BiLSTMTaggerModel.get_span_emb(flatted_dse_context_output, flatted_dse_context_output,
                                                                flatted_candidate_dse_starts,
                                                                flatted_candidate_dse_ends,
                                                                self.config, dropout=self.dropout)
        candidate_dse_span_emb = self.dse_reps_drop(torch.relu(self.dse_reps.forward(candidate_dse_span_emb)))
        if self.config.use_cons_labels:
            """concatenate with the constituent label embeddings"""
            assert self.config.mtl_cons
            _, sys_cons_indexes, sys_cons_scores = self.get_sys_cons(
                cons_lstm_out, masks, num_sentences, sent_lengths, max_sent_length)
            _, sys_cons_indexes = sys_cons_scores.max(dim=-1)
            constituent_label_embeddings = self.constituent_label_embedding.forward(sys_cons_indexes)
            candidate_dse_span_emb = torch.cat([candidate_dse_span_emb, constituent_label_embeddings], dim=-1)
        # Get the span ids
        candidate_dse_span_number = candidate_dse_span_emb.size()[0]
        max_candidate_dse_spans_num_per_sentence = candidate_dse_mask.size()[1]
        sparse_dse_indices = (candidate_dse_mask == 1).nonzero()
        sparse_dse_values = torch.arange(0, candidate_dse_span_number)
        candidate_dse_span_ids = \
            torch.sparse.FloatTensor(sparse_dse_indices.t(), sparse_dse_values,
                                     torch.Size([num_sentences, max_candidate_dse_spans_num_per_sentence])).to_dense()
        candidate_dse_span_ids = candidate_dse_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        predict_dict.update({"candidate_dse_starts": candidate_dse_starts, "candidate_dse_ends": candidate_dse_ends})
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_dse_scores = self.get_dse_unary_scores(
            candidate_dse_span_emb, self.config, self.dropout, 1, "argument scores"
        )
        if self.config.joint:
            flatted_candidate_dse_scores = self.generate_span_scores(
                flatted_candidate_dse_scores, dse_starts_scores, dse_ends_scores,
                flatted_candidate_dse_mask
            )
        candidate_dse_scores = flatted_candidate_dse_scores.index_select(0, candidate_dse_span_ids.view(-1)) \
            .view(candidate_dse_span_ids.size()[0], candidate_dse_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_dse_mask = candidate_dse_mask.type(torch.Tensor).unsqueeze(2).expand(-1, -1, 2).cuda()
        candidate_dse_scores = candidate_dse_scores * spans_dse_mask
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted dse
        predicted_dse_index = \
            self.get_candidate_argument_index(candidate_dse_scores,
                                              candidate_dse_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 2. eval predicted dse
        dense_gold_dse_index = \
            self.get_gold_dense_argu_index(gold_dse, max_sent_length,
                                           candidate_dse_mask.type(torch.LongTensor).view(num_sentences, -1))
        matched_dse_num, sys_dse_num, gold_dse_num = \
            self.eval_predicted_arguments(predicted_dse_index, dense_gold_dse_index, candidate_dse_mask)
        predict_dict.update({"matched_dse_num": matched_dse_num,
                             "sys_dse_num": sys_dse_num,
                             "gold_dse_num": gold_dse_num})
        # 3. compute dse loss
        dse_loss = self.get_argument_focal_loss(
            candidate_dse_scores, dense_gold_dse_index,
            candidate_dse_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 4. get the predicted argument representations according to the index
        if self.use_gold_predicates:
            dse_emb, dse_scores, dse_starts, dse_ends, num_dse = \
                self.get_arguments_according_to_index(candidate_dse_span_emb, candidate_dse_scores,
                                                      candidate_dse_starts, candidate_dse_ends, candidate_dse_span_ids,
                                                      dense_gold_dse_index)
        else:
            dse_emb, dse_scores, dse_starts, dse_ends, num_dse = \
                self.get_arguments_according_to_index(candidate_dse_span_emb, candidate_dse_scores,
                                                      candidate_dse_starts, candidate_dse_ends, candidate_dse_span_ids,
                                                      predicted_dse_index)

        # ARG starts and ends prediction
        # 1. compute the candiate argument starts and ends
        gold_arg_starts, gold_arg_ends, gold_arg_num, gold_arg_starts_one_hot, gold_arg_ends_one_hot = gold_arg

        arg_starts_scores = self.arg_start_scorer.forward(lstm_out)
        arg_ends_scores = self.arg_end_scorer.forward(lstm_out)
        # 2. get the arg starts and ends with probability threshold
        true_pred_starts, predicted_arg_starts_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_starts_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        true_pred_ends, predicted_arg_ends_mask = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_ends_scores, self.config.arg_boundary_prob_threshold, mask=masks)
        # 3. compute the losses of the candidate argument starts and ends
        arg_starts_loss = self.get_arg_boundary_fl_loss(arg_starts_scores, gold_arg_starts_one_hot, mask=masks)
        arg_ends_loss = self.get_arg_boundary_fl_loss(arg_ends_scores, gold_arg_ends_one_hot, mask=masks)
        # 4. evaluate the predicted argument boundary
        num_match_arg_starts, num_pred_arg_starts, num_gold_arg_starts = \
            self.eval_predicted_argument_boundaries(true_pred_starts, gold_arg_starts_one_hot, mask=masks)
        num_match_arg_ends, num_pred_arg_ends, num_gold_arg_ends = \
            self.eval_predicted_argument_boundaries(true_pred_ends, gold_arg_ends_one_hot, mask=masks)
        predict_dict.update({"matched_arg_starts": num_match_arg_starts,
                             "pred_arg_starts": num_pred_arg_starts,
                             "gold_arg_starts": num_gold_arg_starts,

                             "matched_arg_ends": num_match_arg_ends,
                             "pred_arg_ends": num_pred_arg_ends,
                             "gold_arg_ends": num_gold_arg_ends}
                            )
        arg_boundary_mask = None
        # print(self.config.pruning_by_arg_prob)
        if self.config.pruning_by_arg_prob:
            arg_boundary_mask = self.get_arg_boundary_by_threshold(
                arg_starts_scores, arg_ends_scores,
                self.config.arg_boundary_prob_threshold
            )
        """Get the candidate arguments"""
        argu_reps = self.argu_reps_drop_0(torch.relu(self.argu_reps_0.forward(lstm_out)))
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width, arg_boundary_mask)

        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_context_output = BiLSTMTaggerModel.flatten_emb_in_sentence(
            argu_reps, byte_sentence_mask)  # cuda type
        # batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()

        candidate_span_emb = BiLSTMTaggerModel.get_span_emb(flatted_context_output, flatted_context_output,
                                                            flatted_candidate_starts, flatted_candidate_ends,
                                                            self.config, dropout=self.dropout)
        candidate_span_emb = self.argu_reps_drop(torch.relu(self.argu_reps.forward(candidate_span_emb)))
        if self.config.use_cons_labels:
            candidate_span_emb = torch.cat([candidate_span_emb, constituent_label_embeddings], dim=-1)
        # Get the span ids
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number)
        candidate_span_ids = \
            torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                     torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        predict_dict.update({"candidate_starts": candidate_starts, "candidate_ends": candidate_ends})
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_arg_scores = self.get_arg_unary_scores(candidate_span_emb, self.config, self.dropout,
                                                                 1, "argument scores")
        if self.config.joint:
            flatted_candidate_arg_scores = self.generate_span_scores(
                flatted_candidate_arg_scores, arg_starts_scores, arg_ends_scores,
                flatted_candidate_mask
            )
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_mask = candidate_mask.type(torch.Tensor).unsqueeze(2).expand(-1, -1, 2).cuda()
        candidate_arg_scores = candidate_arg_scores * spans_mask
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted arguments
        predicted_arguments_index = \
            self.get_candidate_argument_index(candidate_arg_scores,
                                              candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        if self.config.pruning_by_three_threshold:
            predicted_arguments_index = \
                self.get_candidate_argument_index_by_threshold(
                    candidate_arg_scores, arg_starts_scores, arg_ends_scores,
                    candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1),
                    self.config.arg_three_p_boundary_prob_threshold
                )
        """Model for constituent spans"""
        if self.config.analyze:
            assert self.config.mtl_cons
            _, sys_cons_indexes, _ = self.get_sys_cons(cons_lstm_out, masks,
                                                    num_sentences, sent_lengths, max_sent_length)
            """Add the predicted constituent spans into the ORL arguments"""
            filtered_cons = torch.zeros_like(sys_cons_indexes).type(torch.cuda.BoolTensor)
            for filtered_idx in self.needed_cons_labels_idx:
                booled_sys_cons_index = (sys_cons_indexes == filtered_idx)
                filtered_cons = filtered_cons | booled_sys_cons_index
            # booled_sys_cons_index = sys_cons_indexes > 0
            original_sys_arguments_index = predicted_arguments_index
            predicted_arguments_index = (filtered_cons |
                                         predicted_arguments_index.type(filtered_cons.type())
                                         ).type(torch.cuda.LongTensor)
            pass
        # 2. eval predicted argument
        dense_gold_argus_index = \
            self.get_gold_dense_argu_index(gold_arg, max_sent_length,
                                           candidate_mask.type(torch.LongTensor).view(num_sentences, -1))
        matched_argu_num, sys_argu_num, gold_argu_num = \
            self.eval_predicted_arguments(predicted_arguments_index, dense_gold_argus_index, mask=candidate_mask)
        predict_dict.update({"matched_argu_num": matched_argu_num,
                             "sys_argu_num": sys_argu_num,
                             "gold_argu_num": gold_argu_num})
        # 3. compute argument loss
        argument_loss = self.get_argument_focal_loss(
            candidate_arg_scores, dense_gold_argus_index,
            candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 4. get the predicted argument representations according to the index
        if self.use_gold_arguments:
            arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
                self.get_arguments_according_to_index(candidate_span_emb, candidate_arg_scores,
                                                      candidate_starts, candidate_ends, candidate_span_ids,
                                                      dense_gold_argus_index)
        else:
            arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
                self.get_arguments_according_to_index(candidate_span_emb, candidate_arg_scores,
                                                      candidate_starts, candidate_ends, candidate_span_ids,
                                                      predicted_arguments_index)
        """Model for constituent spans"""
        if self.config.analyze:
            cons_ans, sys_cons_indexes, _ = self.get_sys_cons(cons_lstm_out, masks,
                                                           num_sentences, sent_lengths, max_sent_length,
                                                           arguments_index=predicted_arguments_index)
            sys_arg_emb, sys_arg_scores, sys_arg_starts, sys_arg_ends, sys_num_args = \
                self.get_arguments_according_to_index(candidate_span_emb, candidate_arg_scores,
                                                      candidate_starts, candidate_ends, candidate_span_ids,
                                                      original_sys_arguments_index
                                                      )
            predict_dict.update(cons_ans)
            predict_dict.update(
                {
                    "sys_arg_starts": sys_arg_starts,
                    "sys_arg_ends": sys_arg_ends,
                    "sys_arg_num": sys_num_args,
                }
            )
        """Compute the candidate predicates and arguments semantic roles"""
        srl_labels = self.get_srl_labels(
            dse_starts.cpu(), dse_ends.cpu(), arg_starts.cpu(), arg_ends.cpu(),
            gold_orl, max_sent_length
        )
        srl_scores = self.get_srl_scores(
            arg_emb, dse_emb, self.label_space_size, self.config, self.dropout)
        srl_loss, srl_mask = self.get_srl_softmax_focal_loss(srl_scores, srl_labels, num_args, num_dse)
        # 4. eval the predicted srl
        matched_srl_num, sys_srl_num, gold_srl_num = self.eval_srl(srl_scores, srl_labels, srl_mask)
        predict_dict.update({"matched_srl_num": matched_srl_num,
                             "sys_srl_num": sys_srl_num,
                             "gold_srl_num": gold_srl_num})
        print("sys dse_starts\n", dse_starts)
        print("sys dse_ends\n", dse_ends)
        print("sys arg_starts\n", arg_starts)
        print("sys arg_starts\n", arg_ends)

        predict_dict.update({
            "candidate_arg_scores": candidate_arg_scores,
            "candidate_dse_scores": candidate_dse_scores,
            "dse_starts": dse_starts.cpu(),
            "dse_ends": dse_ends.cpu(),
            "arg_starts": arg_starts.cpu(),
            "arg_ends": arg_ends.cpu(),
            "batch_size": num_sentences,
            "dse_scores": dse_scores,
            "num_args": num_args,
            "num_dse": num_dse,
            "max_num_args": arg_emb.size(1),
            "max_num_dse": dse_emb.size(1),
            "arg_labels": torch.max(srl_scores, 1)[1],  # [num_sentences, num_args, num_preds]
            "srl_scores": srl_scores
        })
        if self.config.mtl:
            return predict_dict, (dse_starts_loss + dse_ends_loss) + dse_loss + \
                   (arg_starts_loss + arg_ends_loss) + argument_loss + srl_loss
        else:
            return predict_dict, dse_loss + argument_loss + srl_loss

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        model_params = torch.load(filepath)
        for model_param, pretrained_model_param in zip(self.parameters(), model_params.items()):
            if len(pretrained_model_param[1].size()) > 1 and pretrained_model_param[1].size()[0] > 5000:  # pretrained word embedding
                pretrained_word_embedding_size = pretrained_model_param[1].size()[0]
                model_param.data[:pretrained_word_embedding_size].copy_(pretrained_model_param[1])
                print("Load {} pretrained word embedding!".format(pretrained_word_embedding_size))
            else:
                model_param.data.copy_(pretrained_model_param[1])
        print('Loaded model from: {}'.format(filepath))
