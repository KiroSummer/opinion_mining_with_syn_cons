import nltk
import sys
import numpy as np
import random

from .dictionary import Dictionary
from collections import OrderedDict
from nltk.tree import Tree
from .constants import PADDING_TOKEN, UNKNOWN_TOKEN
# from .reader import list_of_words_to_ids


PREFIX = "--PTB-CONS-LABEL--"


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


class constituent_tree():
    def __init__(self, sentence, words, tree):
        self.sentence = sentence
        self.words = words
        self.tree = tree
        self.heads = None
        self.nodes = None
        self.indicator = []  # 0 no terminal, 1 terminal
        self.word_position = []
        self.node_idx = []
        self.node_char_idx = []

        self.sentence_length = len(words)
        self.input_length = -1
        self.sentence_index = -1

    def pos(self):
        """[('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]"""
        return self.tree.pos()

    def traverse_tree(self, tree, nodes, indicator, heads, parent, pos, label, word_embeddings):
        # print(tree)
        # print("subtree", subtree)
        if tree.height() > 2:
            subtree_label = PREFIX + tree.label()
            label.add(subtree_label)
            constituent_tree.add_unknown_labels(subtree_label, word_embeddings)
            nodes.append(subtree_label)
            indicator.append(0)
            heads.append(parent - 1)
        else:
            # print("YY", subtree)
            pos.add(tree.label())
            subtree_pos = tree[0]  # word
            subtree_pos = constituent_tree.add_word(subtree_pos, label, word_embeddings)
            nodes.append(subtree_pos)
            indicator.append(1)
            idx = len(nodes) - 1
            self.word_position.append(idx)
            heads.append(parent - 1)
        if tree.height() <= 2:
            return
        parent = len(nodes)
        for i, subtree in enumerate(tree):
            self.traverse_tree(subtree, nodes, indicator, heads, parent, pos, label, word_embeddings)

    @staticmethod
    def add_unknown_labels(label, word_embeddings):
        if label not in word_embeddings:
            embedding_size = len(word_embeddings[PADDING_TOKEN])
            word_embeddings[label] = np.asarray([random.gauss(0, 0.01) for _ in range(embedding_size)])

    @staticmethod
    def add_word(word, word_dict, word_embeddings):
        if word not in word_embeddings:
            word = UNKNOWN_TOKEN
        idx = word_dict.add(word)
        return word

    @staticmethod
    def get_node_char_idx(words, char_dict, lowercase=False):
        max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
        single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int)
        for i, word in enumerate(words):
            single_sample_char_tokens[i, :len(word)] = list_of_words_to_ids(word, char_dict, lowercase)
        return single_sample_char_tokens

    def generate_adjacent(self, pos, label_dict, char_dict, word_embeddings):
        assert self.heads is None
        root_label = PREFIX + self.tree.label()
        nodes, heads = [], []  # TODO notice
        self.traverse_tree(self.tree, nodes, self.indicator, heads,
                           len(heads), pos, label_dict, word_embeddings)
        self.nodes = nodes
        self.heads = heads
        self.input_length = len(self.nodes)
        self.sentence_index = self.input_length - self.sentence_length - 1
        self.node_idx = [label_dict.get_index(node) for node in self.nodes]

        max_word_length = max([len(w) for w in self.nodes] + [3, 4, 5])  # compare with character cnn filter width
        self.node_char_idx = np.zeros([len(self.nodes), max_word_length], dtype=np.int64)
        for i, word in enumerate(self.nodes):
            self.node_char_idx[i, :len(word)] = list_of_words_to_ids(word, char_dict)

        self.node_char_idx = constituent_tree.get_node_char_idx(self.nodes, char_dict)


def load_constituent_trees(file_path, word_dict, char_dict, word_embeddings):
    data = []
    with open(file_path, 'r') as input_file:
        sentence = ""
        for line in input_file.readlines():
            if line.strip() == "":
                data.append(sentence)
                sentence = ""
                continue
            line = line.strip()
            if ' ' not in line:  # avoid the split of leave node of it's PoS
                line = ' ' + line
            sentence += line
        print("Read {} sentence from {}".format(len(data), file_path))

    cons_trees = OrderedDict()
    for sentence in data:
        tree = Tree.fromstring(sentence)
        words = tree.leaves()
        sentence = ' '.join(words)
        cons_trees[sentence] = constituent_tree(sentence, words, tree)

    pos_dict = Dictionary(padding_token=PADDING_TOKEN)
    for sen in cons_trees:
        tree = cons_trees[sen]
        tree.generate_adjacent(pos_dict, word_dict, char_dict, word_embeddings)

    return cons_trees, pos_dict


