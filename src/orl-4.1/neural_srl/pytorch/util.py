import numpy as np
import torch
from torch.autograd import Variable


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal_(param)
            weight.append(param)
    return torch.cat(weight)


def batch_data_variable(batch_x, batch_y, batch_lengths, batch_weights):
    batch_size = len(batch_x)  # batch size
    length = max(batch_lengths)

    words = Variable(torch.LongTensor(batch_size, length).zero_(),
                     requires_grad=False)  # padding with 0
    predicates = Variable(torch.LongTensor(batch_size, length).zero_(),
                          requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(),
                     requires_grad=False)
    padding_answers = Variable(torch.LongTensor(batch_size, length).zero_(),
                               requires_grad=False)
    labels, lengths = [], []

    b = 0
    for s_words, s_answer, s_length, s_weights in zip(batch_x, batch_y,
                                                      batch_lengths,
                                                      batch_weights):
        lengths.append(s_length)
        rel = np.zeros((s_length), dtype=np.int32)
        for i in range(s_length):
            words[b, i] = s_words[1][i]  # word
            predicates[b, i] = s_words[2][i]  # predicate
            rel[i] = s_answer[0][i]
            padding_answers[b, i] = s_answer[0][i]
            masks[b, i] = 1

        # sentence_id = s_words[0][0]  # get the dep_labels_ids of each sentence
        b += 1
        labels.append(rel)

    return words, predicates, labels, torch.LongTensor(
        lengths), masks, padding_answers
