import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from transformers import BertTokenizer


class ScalarMix(torch.nn.Module):
    def __init__(self, mixture_size=4):
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        self.scalar_parameters = Parameter(torch.ones(mixture_size))
        self.gamma = Parameter(torch.tensor(1.0))

    def forward(self, layers):
        normed_weights = F.softmax(self.scalar_parameters, dim=0)
        return self.gamma * sum(
            weight * tensor for weight, tensor in zip(normed_weights, layers)
        )


class Bert_Embedding(nn.Module):
    def __init__(self, bert_path, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding, self).__init__()
        self.bert_layer = bert_layer
        self.bert = BertModel.from_pretrained(bert_path, output_hidden_states=True)
        print(self.bert.config)
        self.scalar_mix = ScalarMix(bert_layer)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks, text_masks, subwords_mask):
        self.eval()
        sen_lens = token_starts_masks.sum(dim=1)
        _, _, bert_outs = self.bert(
            subword_idxs,
            attention_mask=subword_masks
        )  # tuple([Batch_size, max_sentence_length, dim])
        bert_outs = bert_outs[-self.bert_layer:]
        bert_outs = self.scalar_mix(bert_outs)
        # bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        # bert_outs = pad_sequence(bert_outs, batch_first=True)
        zeros = bert_outs.new_zeros(*subwords_mask.size(), bert_outs.size(-1))
        zeros.masked_scatter_(subwords_mask.unsqueeze(-1), bert_outs[text_masks])
        subwords_lens = subwords_mask.sum(-1)
        subwords_lens += (subwords_lens == 0).type(subwords_lens.type())  # 0.0 / 0 -> 0.0 / 1
        bert_outs = zeros.sum(2) / subwords_lens.unsqueeze(-1)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_layer, freeze=False, fix_layer_number=None):
        super(Bert_Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, output_hidden_states=True)
        self.bert_layer = bert_layer

        if freeze:
            self.freeze()
        if fix_layer_number is not None:
            self.fix_several_layers(fix_layer_number)

    def forward(self, subword_idxs, subword_masks, token_starts_masks, text_masks, subwords_mask):
        sen_lens = token_starts_masks.sum(dim=1)
        _, _, bert_outs = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
        )
        bert_outs = bert_outs[-1]  # the last layer of BERT outputs
        # bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        zeros = bert_outs.new_zeros(*subwords_mask.size(), bert_outs.size(-1))
        zeros.masked_scatter_(subwords_mask.unsqueeze(-1), bert_outs[text_masks])
        bert_outs = pad_sequence(zeros, batch_first=True)
        subwords_lens = subwords_mask.sum(-1)
        subwords_lens += (subwords_lens == 0).type(subwords_lens.type())  # 0.0 / 0 -> 0.0 / 1
        bert_outs = bert_outs.sum(2) / subwords_lens.unsqueeze(-1)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False

    def fix_several_layers(self, layer_numer):
        fixed_layer_names = ["embeddings"] if layer_numer >= 0 else []
        for i in range(layer_numer):
            fixed_layer_names.append("encoder.layer." + str(i) + '.')
        print("{} will be fixed".format(fixed_layer_names))
        for name, para in self.bert.named_parameters():
            for layer_name in fixed_layer_names:
                if layer_name in name:
                    para.requires_grad = False
                    break


class Vocab(object):
    def __init__(self, bert_vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_vocab_path, do_lower_case=False
        )

    def numericalize(self, seqs, training=True):
        subwords, masks, starts = [], [], []
        text_masks, subwords_mask = [], []

        for seq in seqs:
            seq = [self.tokenizer.tokenize(token) for token in seq]
            seq = [piece if piece else ["[PAD]"] for piece in seq]
            seq = [["[CLS]"]] + seq + [["[SEP]"]]
            lengths = [0] + [len(piece) for piece in seq]
            # flatten the word pieces
            tokens = sum(seq, [])
            # subwords indexes
            token_idx = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            subwords.append(token_idx)

            # subword masks
            mask = torch.ones(len(tokens), dtype=torch.bool)
            masks.append(mask)
            # subword text mask
            text_mask = torch.BoolTensor([0] + [1] * (len(tokens) - 2) + [0])
            text_masks.append(text_mask)

            # record the start position of all words
            start_idxs = torch.tensor(lengths).cumsum(0)[1:-2]  # bos:0 eos:-2
            # subword start masks
            start_mask = torch.zeros(len(tokens), dtype=torch.bool)
            start_mask[start_idxs] = 1
            starts.append(start_mask)

            # record the start and last position of all words
            start_end_idxs = torch.tensor(lengths).cumsum(0)[1:-1]
            subword_mask = [torch.ones(start_end_idxs[i + 1] - start_end_idxs[i])
                            for i in range(len(start_end_idxs) - 1)]
            subword_mask = pad_sequence(subword_mask, batch_first=True)
            subwords_mask.append(subword_mask)
        max_subword_length = max(m.size(-1) for m in subwords_mask)
        max_sentence_length = max(m.size(0) for m in subwords_mask)
        subwords_mask = [F.pad(mask, (0, max_subword_length - mask.size(1), 0, max_sentence_length - mask.size(0)))
                         for mask in subwords_mask]  # [left, right, top, down]
        subwords_mask = torch.stack(subwords_mask)
        return subwords, masks, starts, text_masks, subwords_mask


class BERT_input(nn.Module):
    def __init__(self, bert_vocab_path, bert_path, bert_layer, bert_dim):
        super(BERT_input, self).__init__()
        self.vocab = Vocab(bert_vocab_path)
        self.bert_input = Bert_Embedding(bert_path, bert_layer, bert_dim)

    def forward(self, seqs):
        subwords, masks, starts, text_masks, subwords_mask = self.vocab.numericalize(seqs)
        subwords = pad_sequence(subwords, batch_first=True).cuda()
        masks = pad_sequence(masks, batch_first=True).cuda()
        starts = pad_sequence(starts, batch_first=True).cuda()
        text_masks = pad_sequence(text_masks, batch_first=True).type(torch.BoolTensor).cuda()
        subwords_mask = subwords_mask.type(torch.BoolTensor).cuda()
        bert_outs = self.bert_input.forward(subwords, masks, starts, text_masks, subwords_mask)
        return bert_outs


class BERT_model(nn.Module):
    def __init__(self, bert_vocab_path, bert_path, bert_layer, bert_dim, fix_layer_number=None):
        super(BERT_model, self).__init__()
        self.vocab = Vocab(bert_vocab_path)
        self.bert_encoder = Bert_Encoder(bert_path, bert_layer,
                                         freeze=False, fix_layer_number=fix_layer_number)

    def forward(self, seqs):
        subwords, masks, starts, text_masks, subwords_mask = self.vocab.numericalize(seqs)
        subwords = pad_sequence(subwords, batch_first=True).cuda()
        masks = pad_sequence(masks, batch_first=True).cuda()
        starts = pad_sequence(starts, batch_first=True).type(torch.BoolTensor).cuda()
        text_masks = pad_sequence(text_masks, batch_first=True).type(torch.BoolTensor).cuda()
        subwords_mask = subwords_mask.type(torch.BoolTensor).cuda()
        bert_outs = self.bert_encoder.forward(subwords, masks, starts, text_masks, subwords_mask)
        return bert_outs
