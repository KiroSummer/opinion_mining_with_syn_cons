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

        self.heads = []
        self.non_terminal_nodes = []  # cons labels, e.g., NP, VP
        self.terminal_nodes = []  # words
        self.indicator = []  # 1 no terminal, 2 terminal

        self.non_terminal_nodes_idx = []
        self.non_terminal_nodes_char_idx = []
        self.terminal_node_idx = []
        self.terminal_node_char_idx = []

        self.sentence_length = len(words)
        self.input_length = -1
        self.sentence_index = -1

    def pos(self):
        """[('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]"""
        return self.tree.pos()

    def traverse_tree(self, tree,
                      non_terminal_nodes, terminal_nodes,
                      non_terminal_nodes_idx, terminal_nodes_idx,
                      indicator,
                      heads,
                      parent,
                      non_terminal_dict, word_dict, pos,
                      word_embeddings):
        # print(tree)
        # print("subtree", subtree)
        if tree.height() > 2:
            non_terminal = tree.label()

            non_terminal_nodes.append(non_terminal)
            non_terminal_nodes_idx.append(non_terminal_dict.add(non_terminal))
            indicator.append(1)
            heads.append(parent - 1)
        else:
            # print("YY", subtree)
            terminal = tree[0]  # word
            terminal_nodes.append(terminal)
            terminal_nodes_idx.append(
                constituent_tree.add_word(terminal, word_dict, word_embeddings)
            )
            indicator.append(2)

            pos.add(tree.label())
            heads.append(parent - 1)
        if tree.height() <= 2:  # 2 == ["V", Tree("Chased")]
            return
        parent = len(non_terminal_nodes) + len(terminal_nodes)
        for i, subtree in enumerate(tree):
            self.traverse_tree(subtree,
                               non_terminal_nodes, terminal_nodes,
                               non_terminal_nodes_idx, terminal_nodes_idx,
                               indicator,
                               heads, parent,
                               non_terminal_dict, word_dict, pos,
                               word_embeddings)

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
        return idx

    @staticmethod
    def get_node_char_idx(words, char_dict, lowercase=False):
        max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
        single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int)
        for i, word in enumerate(words):
            single_sample_char_tokens[i, :len(word)] = list_of_words_to_ids(word, char_dict, lowercase)
        return single_sample_char_tokens

    def generate_adjacent(self, non_terminal_dict, word_dict, char_dict, pos, word_embeddings):
        root_label = self.tree.label()
        self.traverse_tree(self.tree,
                           self.non_terminal_nodes, self.terminal_nodes,
                           self.non_terminal_nodes_idx, self.terminal_node_idx,
                           self.indicator,
                           self.heads, len(self.heads),
                           non_terminal_dict, word_dict, pos,
                           word_embeddings
                           )
        self.input_length = len(self.non_terminal_nodes) + len(self.terminal_nodes)
        self.sentence_index = self.input_length - self.sentence_length - 1

        self.non_terminal_nodes_char_idx = constituent_tree.get_node_char_idx(
            self.non_terminal_nodes, char_dict
        )
        self.terminal_node_char_idx = constituent_tree.get_node_char_idx(
            self.terminal_nodes, char_dict
        )


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

    def reset_sentence(sentence):
        for i in range(len(sentence)):
            if sentence[i] in ["[", "]", "(", ")", "{", "}", "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-"]:
                sentence[i] = '-'

    cons_trees = OrderedDict()
    for sentence in data:
        tree = Tree.fromstring(sentence)
        words = tree.leaves()
        reset_sentence(words)
        sentence = ' '.join(words)
        cons_trees[sentence] = constituent_tree(sentence, words, tree)

    pos_dict = Dictionary(padding_token=PADDING_TOKEN)
    non_terminal_dict = Dictionary(padding_token=PADDING_TOKEN)
    for sen in cons_trees:
        tree = cons_trees[sen]
        tree.generate_adjacent(non_terminal_dict, word_dict, char_dict, pos_dict, word_embeddings)

    return cons_trees, non_terminal_dict, pos_dict,


