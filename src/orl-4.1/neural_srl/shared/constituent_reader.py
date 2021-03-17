import json
import codecs
import numpy as np


from sortedcontainers import SortedSet
from .constants import START_MARKER, END_MARKER, UNKNOWN_TOKEN, PADDING_TOKEN, NULL_LABEL
from .dictionary import Dictionary


def list_of_words_to_ids(list_of_words, dictionary, lowercase=False, pretrained_embeddings=None):
    ids = []
    for s in list_of_words:
        # s = s.encode('utf-8')  # unicode -> utf-8
        if s is None:
            ids.append(-1)
            continue
        if lowercase:
            s = s.lower()
        if (pretrained_embeddings is not None) and (s not in pretrained_embeddings):
            s = UNKNOWN_TOKEN
        ids.append(dictionary.add(s))
    return ids


class constituent_sentence():
    def __init__(self, obj):
        self.sentence = obj["sentence"]
        self.constituent_spans = obj["constituents"]
        self.max_span_width = 30

    def tokenize_cons_spans(self, dictionary, max_cons_width=60):
        cons_span = []
        set_cons_span = set()
        for cons_s in self.constituent_spans:  # remove self-loop V-V
            cons_start, cons_end, cons_label = cons_s
            if cons_label in ["TOP", "S"]:  # todo: add some constrains here
                continue
            if cons_end - cons_start + 1 >= max_cons_width:
                continue
            if (cons_start, cons_end) not in set_cons_span:
                set_cons_span.add((cons_start, cons_end))
                cons_span.append([int(cons_start), int(cons_end), int(dictionary.add(cons_label))])
            else:
                # print("duplicate span of", (cons_start, cons_end, cons_label), '\n', self.sentence)
                pass
        if len(cons_span) == 0:  # if the sentence has no arguments.
            return [[], [], []]
        tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = \
            zip(*cons_span)
        return tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels


def read_constituent_file(file_path):
    sentences = []
    with codecs.open(file_path, encoding="utf8") as f:
        for line in f.readlines():
            sen = json.loads(line)
            cons_sen = constituent_sentence(sen)
            sentences.append(cons_sen)
    print("{} total constituent sentences number {}".format(file_path, len(sentences)))
    return sentences


def tokenize_cons_data(samples, word_dict, char_dict, label_dict, lowercase=False, pretrained_word_embedding=False):
    sample_word_tokens = [list_of_words_to_ids(
        sent.sentence, word_dict, lowercase, pretrained_word_embedding) for sent in samples]
    # for the character
    sample_char_tokens = []
    for sent in samples:
        words = sent.sentence
        max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
        single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int)
        for i, word in enumerate(words):
            single_sample_char_tokens[i, :len(word)] = list_of_words_to_ids(word, char_dict, lowercase)
        # Add the sample char tokens into the sample_char_tokens
        sample_char_tokens.append(single_sample_char_tokens)
    sample_lengths = [len(sent.sentence)for sent in samples]
    sample_cons_span_tokens = [sent.tokenize_cons_spans(label_dict) for sent in samples]
    return list(zip(sample_lengths, sample_word_tokens, sample_char_tokens, sample_cons_span_tokens))


def get_constituent_data(config, file_path, word_dict=None, char_dict=None, word_embeddings=None):
    raw_cons_sentences = read_constituent_file(file_path)
    cons_label_dict = Dictionary()
    cons_label_dict.set_unknown_token(NULL_LABEL)

    # tokenized the data
    if word_dict.accept_new is False:
        word_dict.accept_new = True
    if char_dict.accept_new is False:
        char_dict.accept_new = True
    cons_samples = tokenize_cons_data(raw_cons_sentences, word_dict, char_dict, cons_label_dict,
                                      False, word_embeddings)
    # word_dict.accept_new = False
    # char_dict.accept_new = False
    # cons_label_dict.accept_new = False

    print("="*10, "Constituent Info", "="*10)
    print("Extract {} tags".format(cons_label_dict.size()))
    # print("Extract {} words and {} tags".format(word_dict.size(), cons_label_dict.size()))
    print("Max sentence length: {}".format(max([s[0] for s in cons_samples])))
    return cons_samples, word_dict, char_dict, cons_label_dict
