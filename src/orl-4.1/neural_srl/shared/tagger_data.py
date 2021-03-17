from .constants import UNKNOWN_TOKEN
from .constants import PADDING_TOKEN

import numpy as np
import random
import torch
import math


def tensorize(sentence, max_length):
    """ Input:
      - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
            s1 is always a sequence of word ids.
            sk is always a sequence of label ids.
            s2 ... sk-1 are sequences of feature ids,
              such as predicate or supertag features.
      - max_length: The maximum length of sequences, used for padding.
  """
    x = np.array([t for t in zip(*sentence[1:3])])
    x = x.transpose()  # SRL: (T, 2) T is the sentence length
    return {"sentence_id": sentence[0], "inputs": x, "chars": sentence[3],
            "labels": sentence[-1], "sentence_length": len(sentence[0])}


class TaggerData(object):
    def __init__(self, config,
                 train_sents, dev_sents, eval_data,
                 cons_data, auto_cons_data,
                 dep_data, auto_dep_data,
                 word_dict, char_dict, pos_dict,
                 label_dict,
                 cons_label_dict, dep_label_dict,
                 embeddings, embeddings_shapes):
        train_sents, dev_sents = list(train_sents), list(dev_sents)
        print("Extract {}, {} training dev".format(len(train_sents), len(dev_sents)))
        self.max_train_length = config.max_train_length
        self.max_dev_length = max([s[0] for s in dev_sents]) if len(dev_sents) > 0 else 0
        self.batch_size = config.batch_size
        self.max_tokens_per_batch = config.max_tokens_per_batch

        self.train_samples = [s for s in train_sents if s[0] <= self.max_train_length]
        self.dev_samples = dev_sents
        self.eval_data = eval_data

        self.cons_data = [s for s in cons_data if s[0] <= self.max_train_length]
        self.auto_cons_data = auto_cons_data
        self.dep_data = dep_data
        self.auto_dep_data = auto_dep_data

        self.word_dict = word_dict
        self.char_dict = char_dict
        self.pos_dict = pos_dict
        self.label_dict = label_dict

        self.cons_label_dict = cons_label_dict
        self.dep_label_dict = dep_label_dict
        self.word_embeddings, self.word_embedding_shapes = None, None

        self.word_padding_id = word_dict.str2idx[PADDING_TOKEN]
        self.word_unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
        self.char_padding_id = char_dict.str2idx[PADDING_TOKEN]
        self.char_unk_id = char_dict.str2idx[UNKNOWN_TOKEN]
        print("Word padding id {}, unk id {}".format(self.word_padding_id, self.word_unk_id))
        print("Char padding id {}, unk id {}".format(self.char_padding_id, self.char_unk_id))
        if embeddings is not None:
            self.word_embeddings = embeddings
        if embeddings_shapes is not None:
            self.word_embedding_shapes = embeddings_shapes

    def tensorize_batch_cons_samples(self, samples):
        """
        tensorize the batch samples
        :param samples: List of samples
        :return: tensorized batch samples
        """
        batch_sample_size = len(samples)
        max_sample_length = max([sam[0] for sam in samples])
        max_sample_word_length = max([sam[2].shape[1] for sam in samples])

        max_sample_cons_number = max([len(sam[3][0]) for sam in samples])
        # input
        padded_sample_lengths = np.zeros(batch_sample_size, dtype=np.int)
        padded_word_tokens = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_char_tokens = np.zeros([batch_sample_size, max_sample_length, max_sample_word_length], dtype=np.int)
        # gold arg
        if max_sample_cons_number == 0:
            max_sample_cons_number = 1
        padded_cons_starts = np.zeros([batch_sample_size, max_sample_cons_number], dtype=np.int)
        padded_cons_ends = np.zeros([batch_sample_size, max_sample_cons_number], dtype=np.int)
        padded_cons_labels = np.zeros([batch_sample_size, max_sample_cons_number], dtype=np.int)
        padded_num_gold_cons = np.zeros(batch_sample_size, dtype=np.int)

        padded_gold_cons_starts_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_gold_cons_ends_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        for i, sample in enumerate(samples):
            sample_length = sample[0]
            padded_sample_lengths[i] = sample_length
            # input
            padded_word_tokens[i, :sample_length] = sample[1]
            padded_char_tokens[i, :sample[2].shape[0], : sample[2].shape[1]] = sample[2]
            # gold cons
            sample_cons_number = len(sample[3][0])
            padded_cons_starts[i, :sample_cons_number] = sample[3][0]
            padded_cons_ends[i, :sample_cons_number] = sample[3][1]
            padded_cons_labels[i, :sample_cons_number] = sample[3][2]
            padded_num_gold_cons[i] = sample_cons_number

            padded_gold_cons_starts_one_hot[i][list(sample[3][0])] = 1
            padded_gold_cons_ends_one_hot[i][list(sample[3][1])] = 1
        return torch.from_numpy(padded_sample_lengths),\
                torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_char_tokens),\
                torch.from_numpy(padded_cons_starts), torch.from_numpy(padded_cons_ends), \
                torch.from_numpy(padded_cons_labels), torch.from_numpy(padded_num_gold_cons), \
                torch.from_numpy(padded_gold_cons_starts_one_hot), torch.from_numpy(padded_gold_cons_ends_one_hot)

    def tensorize_batch_samples(self, samples):
        """
        tensorize the batch samples
        :param samples: List of samples
        :return: tensorized batch samples
        """
        batch_sample_size = len(samples)
        max_sample_length = max([sam[0] for sam in samples])
        max_sample_word_length = max([sam[2].shape[1] for sam in samples])

        max_sample_orl_number = max([len(sam[3][0]) for sam in samples])
        max_sample_dse_number = max([len(sam[4][0]) for sam in samples])
        max_sample_arg_number = max([len(sam[5][0]) for sam in samples])

        # input
        padded_sample_lengths = np.zeros(batch_sample_size, dtype=np.int64)
        padded_word_tokens = np.zeros([batch_sample_size, max_sample_length], dtype=np.int64)
        padded_char_tokens = np.zeros([batch_sample_size, max_sample_length, max_sample_word_length], dtype=np.int64)
        # gold dse
        padded_gold_dses_starts = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
        padded_gold_dses_ends = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
        padded_num_gold_dses = np.zeros(batch_sample_size, dtype=np.int64)
        padded_gold_dses_starts_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_gold_dses_ends_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        # gold arg
        padded_gold_argus_starts = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_gold_argus_ends = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_num_gold_argus = np.zeros(batch_sample_size, dtype=np.int64)
        padded_gold_argus_starts_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_gold_argus_ends_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)

        # if max_sample_arg_number == 0:
        #     max_sample_arg_number = 1
        padded_orl_dses_starts = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_dses_ends = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_argus_starts = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_argus_ends = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_labels = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_nums = np.zeros([batch_sample_size], dtype=np.int64)

        sample_words = []
        for i, sample in enumerate(samples):
            sample_length = sample[0]
            padded_sample_lengths[i] = sample_length
            # input
            padded_word_tokens[i, :sample_length] = sample[1]
            padded_char_tokens[i, :sample[2].shape[0], : sample[2].shape[1]] = sample[2]
            # gold dse
            sample_dse_number = len(sample[4][0])
            padded_gold_dses_starts[i, :sample_dse_number] = sample[4][0]
            padded_gold_dses_ends[i, :sample_dse_number] = sample[4][1]
            padded_num_gold_dses[i] = sample_dse_number
            padded_gold_dses_starts_one_hot[i][list(sample[4][0])] = 1
            padded_gold_dses_ends_one_hot[i][list(sample[4][1])] = 1
            # gold arg
            sample_arg_number = len(sample[5][0])
            padded_gold_argus_starts[i, :sample_arg_number] = sample[5][0]
            padded_gold_argus_ends[i, :sample_arg_number] = sample[5][1]
            padded_num_gold_argus[i] = sample_arg_number
            padded_gold_argus_starts_one_hot[i][list(sample[5][0])] = 1
            padded_gold_argus_ends_one_hot[i][list(sample[5][1])] = 1
            # output
            sample_orl_number = len(sample[3][0])
            padded_orl_dses_starts[i, :sample_orl_number] = sample[3][0]
            padded_orl_dses_ends[i, :sample_orl_number] = sample[3][1]
            padded_orl_argus_starts[i, :sample_orl_number] = sample[3][2]
            padded_orl_argus_ends[i, :sample_orl_number] = sample[3][3]
            padded_orl_labels[i, :sample_orl_number] = sample[3][4]
            padded_orl_nums[i] = sample_orl_number

            sample_words.append(sample[-1])
        return torch.from_numpy(padded_sample_lengths),\
                torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_char_tokens),\
                torch.from_numpy(padded_gold_dses_starts), torch.from_numpy(padded_gold_dses_ends),\
                torch.from_numpy(padded_num_gold_dses),\
                torch.from_numpy(padded_gold_dses_starts_one_hot), torch.from_numpy(padded_gold_dses_ends_one_hot),\
                torch.from_numpy(padded_gold_argus_starts), torch.from_numpy(padded_gold_argus_ends),\
                torch.from_numpy(padded_num_gold_argus),\
                torch.from_numpy(padded_gold_argus_starts_one_hot), torch.from_numpy(padded_gold_argus_ends_one_hot),\
                torch.from_numpy(padded_orl_dses_starts), torch.from_numpy(padded_orl_dses_ends),\
                torch.from_numpy(padded_orl_argus_starts), torch.from_numpy(padded_orl_argus_ends),\
                torch.from_numpy(padded_orl_labels), torch.from_numpy(padded_orl_nums), \
               sample_words

    def tensorize_dep_samples(self, samples):
        batch_num_samples = len(samples)
        max_sample_length = max([len(sam[0]) for sam in samples])
        max_sample_word_length = max([sam[1].shape[1] for sam in samples])
        # input
        lengths = np.zeros(batch_num_samples, dtype=np.int)
        padded_word_tokens = np.zeros([batch_num_samples, max_sample_length], dtype=np.int64)
        padded_char_tokens = np.zeros([batch_num_samples, max_sample_length, max_sample_word_length], dtype=np.int64)
        masks = np.zeros([batch_num_samples, max_sample_length], dtype=np.int64)
        heads, labels = [], []
        for i, sample in enumerate(samples):
            length = len(sample[0])
            lengths[i] = length
            # input
            padded_word_tokens[i, :length] = sample[0]
            padded_char_tokens[i, :sample[1].shape[0], :sample[1].shape[1]] = sample[1]
            masks[i, :length] = np.ones((length), dtype=np.int64)
            # output
            heads.append(torch.from_numpy(sample[2]))
            labels.append(torch.from_numpy(sample[3]))
        return torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_char_tokens), torch.from_numpy(masks), \
               torch.from_numpy(lengths), heads, labels

    def get_dep_training_data(self, include_last_batch=False):
        """
        :param include_last_batch:
        :return:
        """
        random.shuffle(self.dep_data)
        results, batch_tensors = [], []
        batch_num_tokens = 0
        for i, example in enumerate(self.dep_data):
            num_words = len(example[0])
            if len(batch_tensors) >= self.batch_size or batch_num_tokens + num_words >= self.max_tokens_per_batch:
                results.append(batch_tensors)
                batch_tensors = []
                batch_num_tokens = 0
            batch_tensors.append(example)
            batch_num_tokens += num_words

        results = [self.tensorize_dep_samples(batch_sample) for i, batch_sample in enumerate(results)]
        print("Extracted {} samples and {} batches.".format(len(self.dep_data), len(results)))
        return results

    def get_cons_data(self, include_last_batch=False, subbatch_size=1):
        """ Get shuffled training samples. Called at the beginning of each epoch.
        """
        # TODO: Speed up: Use variable size batches (different max length).
        random.shuffle(self.cons_data)  # TODO when the model is stable, uncomment it
        assert include_last_batch is True

        num_samples = len(self.cons_data)
        print("Total {} constituent samples".format(num_samples))
        results, batch_tensors = [], []
        batch_num_tokens = 0
        for i, example in enumerate(self.cons_data):
            num_words = example[0]
            if len(batch_tensors) >= self.batch_size or batch_num_tokens + num_words >= self.max_tokens_per_batch:
                results.append(batch_tensors)
                batch_tensors = []
                batch_num_tokens = 0
            batch_tensors.append(example)
            batch_num_tokens += num_words

        results = [self.tensorize_batch_cons_samples(batch_sample) for i, batch_sample in enumerate(results)]
        print("Extracted {} constituent samples and {} batches.".format(num_samples, len(results)))
        # num_sub_batches = len(results) // subbatch_size
        # subbatches_data = []
        # for i in range(num_sub_batches):
        #     subbatches_data.append(results[i * subbatch_size: (i + 1) * subbatch_size])
        # print("Extracted {} constituent samples and {} sub batches.".format(num_samples, len(subbatches_data)))
        return results

    def get_training_data(self, include_last_batch=False):
        """ Get shuffled training samples. Called at the beginning of each epoch.
        """
        # TODO: Speed up: Use variable size batches (different max length).
        #random.shuffle(self.train_samples)  # TODO when the model is stable, uncomment it
        assert include_last_batch is True

        num_samples = len(self.train_samples)
        print("Total {} training samples".format(num_samples))
        results, batch_tensors = [], []
        batch_num_tokens = 0
        for i, example in enumerate(self.train_samples):
            num_words = example[0]
            if len(batch_tensors) >= self.batch_size or batch_num_tokens + num_words >= self.max_tokens_per_batch:
                results.append(batch_tensors)
                batch_tensors = []
                batch_num_tokens = 0
            batch_tensors.append(example)
            batch_num_tokens += num_words

        results = [self.tensorize_batch_samples(batch_sample) for i, batch_sample in enumerate(results)]
        print("Extracted {} samples and {} batches.".format(num_samples, len(results)))
        return results

    def get_development_data(self, batch_size=None):
        if batch_size is None:
            return self.dev_samples

        num_samples = len(self.dev_samples)
        batched_tensors = [self.dev_samples[i: min(i + batch_size, num_samples)]
                           for i in range(0, num_samples, batch_size)]
        results = [self.tensorize_batch_samples(t) for t in batched_tensors]
        return results

    def get_test_data(self, test_sentences, batch_size=None):
        num_samples = len(test_sentences)
        batched_tensors = [test_sentences[i: min(i + batch_size, num_samples)]
                           for i in range(0, num_samples, batch_size)]
        if batch_size is None:
            return batched_tensors
        results = [self.tensorize_batch_samples(t) for t in batched_tensors]
        return results


def mix_training_data(datas):
    nums = [len(data) for data in datas]
    max_num = max(nums)
    expanded_nums = [int(math.ceil(max_num / n)) for n in nums]

    results = list(zip(*[datas[i] * e_n for i, e_n in enumerate(expanded_nums)]))
    print("After mixture, total {} batched samples".format(len(results)))
    return results
