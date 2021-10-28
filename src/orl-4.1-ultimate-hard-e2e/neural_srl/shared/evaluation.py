''' Framework independent evaluator. Not in use yet.
'''
import numpy
import os
from os.path import join
# import subprocess
from .constants import ROOT_DIR
from .conll_utils import print_gold_to_conll
# from .measurements import Timer


class TaggerEvaluator(object):
    def __init__(self, data):
        self.data = data
        self.best_accuracy = 0.0
        self.has_best = False

    def compute_accuracy(self, predictions):
        for x, y in zip(predictions,
                        [sent[2] for sent in self.data
                         ]):  # the predication's order should be the origin
            assert len(x) == y
        predictions = numpy.concatenate(predictions)
        tensors = self.data
        answer = numpy.concatenate(
            [sent[1].reshape(sent[1].shape[1]) for sent in tensors])
        # predictions.resize(predictions.shape[0])  # resize the answer to the [length, 1]
        num_correct = numpy.equal(predictions, answer).sum()
        num_total = answer.shape[0]
        self.accuracy = (100.0 * num_correct) / num_total
        print("Accuracy: {:.3f} ({}/{})".format(self.accuracy, num_correct,
                                                num_total))

    def evaluate(self, predictions):
        self.compute_accuracy(predictions)
        self.has_best = self.accuracy > self.best_accuracy
        if self.has_best:
            print("Best accuracy so far: {:.3f}".format(self.accuracy))
            self.best_accuracy = self.accuracy


class PropIdEvaluator(object):
    def __init__(self, data, label_dict, target_label='V',
                 use_se_marker=False):
        self.data = data
        self.label_dict = label_dict
        self.target_label_id = label_dict.str2idx[target_label]
        self.best_accuracy = 0.0
        self.has_best = False

    def compute_accuracy(self, predictions):
        _, y, _, weights = self.data
        # print predictions.shape, predictions
        identified = numpy.equal(predictions, self.target_label_id)
        print(y)
        # print self.target_label_id
        # print identified
        # exit()
        num_correct = numpy.sum(
            numpy.logical_and(numpy.equal(predictions, y), identified) * weights)
        num_identified = numpy.sum(identified * weights)
        num_gold = numpy.sum(numpy.equal(y, self.target_label_id) * weights)
        self.precision = 100.0 * num_correct / num_identified
        self.recall = 100.0 * num_correct / num_gold
        self.accuracy = 2 * self.precision * self.recall / (self.precision + self.recall)
        print("Accuracy: {:.3f} ({:.3f}, {:.3f})".format(
            self.accuracy, self.precision, self.recall))

    def evaluate(self, predictions):
        self.compute_accuracy(predictions)
        self.has_best = self.accuracy > self.best_accuracy
        if self.has_best:
            print("Best accuracy so far: {:.3f}".format(self.accuracy))
            self.best_accuracy = self.accuracy


class SRLEvaluator(TaggerEvaluator):
    def __init__(self):
        self.best_accuracy = -1.0
        self.has_best = False

    def compute_accuracy(self, predictions):
        print("exit()")
        exit()
