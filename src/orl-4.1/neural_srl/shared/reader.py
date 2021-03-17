import json
import codecs
import random
import numpy as np
from sortedcontainers import SortedSet

from .constants import START_MARKER, END_MARKER, UNKNOWN_TOKEN, PADDING_TOKEN, NULL_LABEL
from .dictionary import Dictionary
from .constituent_reader import get_constituent_data
from .syntactic_extraction import SyntacticCONLL, load_dependency_trees
from .constituent_extraction import load_constituent_trees
from .measurements import Timer


def get_sentences(filepath, use_se_marker=False):
    """ Read tokenized sentences from file """
    sentences = []
    with open(filepath) as f:
        for line in f.readlines():
            inputs = line.strip().split('|||')
            lefthand_input = inputs[0].strip().split()
            # If gold tags are not provided, create a sequence of dummy tags.
            righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                else ['O' for _ in lefthand_input]
            if use_se_marker:
                words = [START_MARKER] + lefthand_input + [END_MARKER]
                labels = [None] + righthand_input + [None]
            else:
                words = lefthand_input
                labels = righthand_input
            sentences.append((words, labels))
    return sentences


class srl_sentence():
    def __init__(self, obj):
        self.sentences = obj["sentences"]
        self.srl = obj["orl"]
        self.dse = []
        self.argus = []
        self.max_arg_length = 30

    def get_labels(self, dictionary):
        ids = []
        for s in self.srl:
            s = s[-1]
            if s is None:
                ids.append(-1)
                continue
            ids.append(dictionary.add(s))
        return ids

    def tokenize_argument_spans(self, dictionary):
        srl_span = []
        for srl_label in self.srl:  # remove self-loop V-V
            # print(srl_label)
            if srl_label[-1] == "DSE":
                self.dse.append(srl_label)
                # dictionary.add(srl_label[-1])
            else:
                self.argus.append(srl_label)
                srl_span.append([int(srl_label[0]), int(srl_label[1]),
                                 int(srl_label[2]), int(srl_label[3]),
                                 int(dictionary.add(srl_label[4]))])
        if len(srl_span) == 0:  # if the sentence has no arguments.
            return [[], [], [], [], []]
        tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = \
            zip(*srl_span)
        tmp = list(zip(tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends,
                       tokenized_arg_labels))
        orl_span = list(SortedSet(tmp))  # remove the same [dse_start, dse_end, x_start, x_end, label]
        tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = \
            zip(*orl_span)
        return tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels

    def tokenize_dse_spans(self):
        dse_span = []
        for dse in self.dse:
            dse_start, dse_end, _, _, dse_label = dse
            assert dse_label == "DSE"
            dse_span.append([int(dse_start), int(dse_end)])
        tokenized_dse_starts, tokenized_dse_ends = zip(*dse_span)
        dse_span = list(zip(tokenized_dse_starts, tokenized_dse_ends))
        dse_span = list(SortedSet(dse_span))
        tokenized_dse_starts, tokenized_dse_ends = zip(*dse_span)
        return tokenized_dse_starts, tokenized_dse_ends

    def tokenize_arg_spans(self):
        arg_span = []
        for arg in self.argus:
            _, _, arg_starts, arg_ends, label = arg
            assert label in ["TARGET", "AGENT"]
            arg_span.append([int(arg_starts), int(arg_ends)])
        if len(arg_span) == 0:
            return [], []
        tokenized_arg_starts, tokenized_arg_ends = zip(*arg_span)
        arg_span = list(zip(tokenized_arg_starts, tokenized_arg_ends))
        arg_span = list(SortedSet(arg_span))
        tokenized_dse_starts, tokenized_dse_ends = zip(*arg_span)
        return tokenized_dse_starts, tokenized_dse_ends

    def get_all_gold_predicates(self):
        if len(self.srl) > 0:
            predicates, _, _, _ = zip(*self.srl)
        else:
            predicates = []
        predicates = np.unique(predicates)
        return predicates, len(predicates)

    def get_all_gold_arguments(self):
        arguments = []
        for srl_label in self.srl:  # remove self-loop V-V
            if srl_label[-1] in ["V", "C-V"]:
                continue
            # if int(srl_label[2]) - int(srl_label[1]) + 1 > self.max_arg_length:
            #     continue
            arguments.append([int(srl_label[1]), int(srl_label[2])])
        if len(arguments) == 0:  # if the sentence has no arguments.
            return [[], []]
        tokenized_arg_starts, tokenized_arg_ends = zip(*arguments)
        # self.check_for_unoverlapping_argument_start_and_end(tokenized_arg_starts, tokenized_arg_ends)
        return tokenized_arg_starts, tokenized_arg_ends

    def check_for_unoverlapping_argument_start_and_end(self, arg_starts, arg_ends):
        "Because the start of an argument can also be the end of itself and can also be ..."
        exit()
        assert len(arg_starts) == len(arg_ends)
        zipped_arg = zip(arg_starts, arg_ends)
        for arg in zipped_arg:
            start, end = arg[0], arg[1]
            if start == end:  # some arg is a single word
                continue
            if start in arg_ends:
                print(start, 'in span end', self.doc_key, self.sentences)


def get_srl_sentences(filepath):
    """
    Data loading with json format.
    """
    sentences = []
    with codecs.open(filepath, encoding="utf8") as f:
        for line in f.readlines():
            sen = json.loads(line)
            srl_sen = srl_sentence(sen)
            sentences.append(srl_sen)
        print("{} total sentences number {}".format(filepath, len(sentences)))
    return sentences


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def get_pretrained_embeddings(filepath):
    embeddings = dict()
    with open(filepath, 'r') as f:
        for line in f:
            info = line.strip().split()
            embeddings[info[0]] = normalize(np.asarray([float(r) for r in info[1:]]))  # normalize the embedding
        f.close()
    embedding_size = len(list(embeddings.values())[0])
    print('{}, embedding size={}'.format(filepath, embedding_size))
    # add START and END MARKER, PADDING and UNKNOWN token in embedding dict
    embeddings[START_MARKER] = np.asarray([random.gauss(0, 0.01) for _ in range(embedding_size)])
    embeddings[END_MARKER] = np.asarray([random.gauss(0, 0.01) for _ in range(embedding_size)])
    # set padding the unknown token id into the embeddings
    if PADDING_TOKEN not in embeddings:
        embeddings[PADDING_TOKEN] = np.zeros(embedding_size)
    if UNKNOWN_TOKEN not in embeddings:
        embeddings[UNKNOWN_TOKEN] = np.zeros(embedding_size)
    return embeddings


def tokenize_data(data, word_dict, char_dict, label_dict, lowercase=False, pretrained_word_embedding=None):
    """
    :param data: the raw input sentences
    :param word_dict: word dictionary
    :param char_dict: character dictionary
    :param label_dict: srl label dictionary
    :param lowercase: bool value, if the word or character needs to lower
    :param pretrained_word_embedding: pre-trained word embedding
    :return: a list storing the [sentence id, length, [words], [heads], [characters], [srl argument spans]]
    """
    sample_sentence_words = [sent.sentences for sent in data]
    sample_word_tokens = [list_of_words_to_ids(sent.sentences, word_dict, lowercase, pretrained_word_embedding)
                          for sent in data]  # sent.sentences[0] is the words of the sentence
    # for the character
    sample_char_tokens = []
    for sent in data:
        words = sent.sentences
        max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
        single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int64)
        for i, word in enumerate(words):
            single_sample_char_tokens[i, :len(word)] = list_of_words_to_ids(word, char_dict, lowercase)
        # Add the sample char tokens into the sample_char_tokens
        sample_char_tokens.append(single_sample_char_tokens)
    sample_lengths = [len(sent.sentences)for sent in data]
    sample_orl_span_tokens = [sent.tokenize_argument_spans(label_dict) for sent in data]
    sample_dse_span_tokens = [sent.tokenize_dse_spans() for sent in data]
    sample_arg_span_tokens = [sent.tokenize_arg_spans() for sent in data]
    # sample_gold_predicates = [sent.get_all_gold_predicates() for sent in data]
    # sample_gold_arguments = [sent.get_all_gold_arguments() for sent in data]
    sample_lengths = np.array(sample_lengths)
    sample_word_tokens = np.array(sample_word_tokens)
    sample_char_tokens = np.array(sample_char_tokens)
    sample_orl_span_tokens = np.array(sample_orl_span_tokens)
    sample_dse_span_tokens = np.array(sample_dse_span_tokens)
    sample_arg_span_tokens = np.array(sample_arg_span_tokens)
    # sample_gold_predicates = np.array(sample_gold_predicates)
    # sample_gold_arguments = np.array(sample_gold_arguments)
    return list(zip(sample_lengths, sample_word_tokens, sample_char_tokens,
                    sample_orl_span_tokens, sample_dse_span_tokens, sample_arg_span_tokens, sample_sentence_words
                    ))


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


def load_eval_data(eval_path):
    eval_data = []
    # with open(eval_path, 'r') as f:
    #     eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    # for doc_id, example in enumerate(eval_examples):
    #     eval_data.extend(split_example_for_eval(example))
    # print("Loaded {} eval examples.".format(len(eval_data)))
    return eval_data


def get_srl_data(config, train_data_path, dev_data_path,
                 cons_path, auto_cons_path,
                 dep_path, auto_dep_path
                 ):
    # Load sentences (documents) from data paths respectively.
    raw_train_sents = get_srl_sentences(train_data_path)
    raw_dev_sents = get_srl_sentences(dev_data_path)
    # Load dev data
    eval_data = load_eval_data(dev_data_path)

    # Prepare word embedding dictionary.
    word_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
    # Prepare char dictionary.
    char_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
    assert config.char_vocab_file != ""
    char_path = config.char_vocab_file
    with open(char_path, 'r') as f_char:
        for line in f_char:
            char_dict.add(line.strip())
        f_char.close()
    char_dict.accept_new = False
    print('Load {} chars, Dictionary freezed.'.format(char_dict.size()))
    # Parpare SRL label dictionary.
    label_dict = Dictionary()
    label_dict.set_unknown_token(NULL_LABEL)  # train corpus contains the label 'O' ?
    # Training data: Get tokens and labels: [sentence_id, word, predicate, label]
    word_dict.accept_new, char_dict.accept_new = True, True

    word_embeddings = get_pretrained_embeddings(config.word_embedding)  # get pre-trained embeddings
    train_samples = tokenize_data(raw_train_sents, word_dict, char_dict, label_dict, False, word_embeddings)
    dev_samples = tokenize_data(raw_dev_sents, word_dict, char_dict, label_dict, False, word_embeddings)
    print("="*10, "ORL data info", "="*10)
    print("Extract {} words and {} tags".format(word_dict.size(), label_dict.size()))
    print("Max training sentence length: {}".format(max([s[0] for s in train_samples])))
    print("Max development sentence length: {}".format(max([s[0] for s in dev_samples])))

    with Timer("Read constituent trees and automatic constituent trees of ORL data."):
        cons_samples, word_dict, char_dict, cons_label_dict = \
            get_constituent_data(config, cons_path, word_dict, char_dict, word_embeddings)

        auto_cons, pos_dict = load_constituent_trees(
            auto_cons_path, word_dict, char_dict, word_embeddings
        )

    with Timer("Read dependency trees and automatic dependency trees of dependency data."):
        dep_label_dict = Dictionary()
        dep_trees = SyntacticCONLL()
        dep_trees.read_from_file(dep_path, prune_ratio=config.dep_prune_ratio)
        dep_trees.tokenize_dep_trees(word_dict, char_dict, dep_label_dict, word_embeddings)

        auto_deps = load_dependency_trees(
            auto_dep_path, word_dict, char_dict, dep_label_dict, word_embeddings
        )

    # set word dict freezed.
    word_dict.accept_new = False

    word_embedding = np.asarray([word_embeddings[w] for w in word_dict.idx2str])
    word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
    print("word embedding shape {}".format(word_embedding_shape))
    return (train_samples, dev_samples, eval_data,
            cons_samples, auto_cons,
            dep_trees.sample_dep_data, auto_deps,
            word_dict, char_dict, pos_dict,
            label_dict,
            cons_label_dict, dep_label_dict,
            word_embedding, word_embedding_shape
            )


def get_srl_test_data(filepath, config, word_dict, char_dict, label_dict,
                      auto_cons_path, auto_dep_path,
                      dep_label_dict=None,
                      allow_new_words=True):
    """get the test data from file"""
    word_dict.accept_new = allow_new_words
    if label_dict.accept_new:
        label_dict.set_unknown_token(NULL_LABEL)
        label_dict.accept_new = False

    if filepath is not None and filepath != '':
        samples = get_srl_sentences(filepath)
    else:
        samples = []

    word_to_embeddings = get_pretrained_embeddings(config.word_embedding)
    test_samples = []
    if allow_new_words:
        test_samples = tokenize_data(samples, word_dict, char_dict, label_dict, False,
                                     word_to_embeddings)
        # tokens = [list_of_words_to_ids(sent[1], word_dict, True, word_to_embeddings) for sent in samples]
    else:
        _ = [list_of_words_to_ids(sent[1], word_dict, True) for sent in samples]

    with Timer("Read constituent trees and automatic constituent trees of ORL data."):
        auto_cons, pos_dict = load_constituent_trees(
            auto_cons_path, word_dict, char_dict, word_to_embeddings
        )

    with Timer("Read dependency trees and automatic dependency trees of dependency data."):
        auto_deps = load_dependency_trees(
            auto_dep_path, word_dict, char_dict, dep_label_dict, word_to_embeddings
        )

    word_embedding = np.asarray([word_to_embeddings[w] for w in word_dict.idx2str])
    word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
    return (test_samples, auto_cons, auto_deps, word_embedding, word_embedding_shape)
