# Evaluation util functions for PropBank SRL.

import codecs
import operator
import os
import subprocess

from collections import Counter, OrderedDict
import json


_SRL_CONLL_EVAL_SCRIPT = "../run_eval.sh"
_SRL_CONLL09_EVAL_SCRIPT = "../run_eval_conll09.sh"


def load_gold_orl_span(eval_path):
    eval_data = []
    with open(eval_path, 'r') as f:
        eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    dses, ans, sentences = [], [], []
    dse_num, arg_num = 0, 0
    arg_total_length = 0
    for sentence in eval_examples:
        dse, orl_spans = [], {}
        for span in sentence["orl"]:
            d_s, d_e, span_start, span_end, span_label = \
                int(span[0]), int(span[1]), int(span[2]), int(span[3]), span[4]
            if span_label == "DSE":
                dse.append((d_s, d_e))
                dse_num += 1
                if (d_s, d_e) not in orl_spans:
                    orl_spans[(d_s, d_e)] = []
            else:
                arg_num += 1
                if (d_s, d_e) not in orl_spans:
                    orl_spans[(d_s, d_e)] = [(span_start, span_end, span_label)]
                else:
                    orl_spans[(d_s, d_e)].append((span_start, span_end, span_label))
                arg_total_length += span_end - span_start + 1
        dses.append(dse)
        ans.append(orl_spans)
        sentences.append(' '.join(sentence["sentences"]))
    print("Extract {} dses from {}".format(dse_num, eval_path))
    print("Extract {} arguments from {}".format(arg_num, eval_path))
    print("Total gold arg length is {}, and the average number of args is {}".format(
        arg_total_length, 1.0 * arg_total_length / arg_num)
    )
    return dses, ans, sentences


def split_example_for_eval(example):
    """Split document-based samples into sentence-based samples for evaluation.
      Args:
        example:
      Returns:
        Tuple of (sentence, list of SRL relations)
    """
    sentences = example["sentences"]
    word_offset = 0
    samples = []
    assert len(sentences) == 1
    for i, sentence in enumerate(sentences):
        assert i == 0  # For CoNLL-2005, there are always document == sentence.
        srl_rels = {}
        ner_spans = []  # Unused.
        for r in example["srl"][i]:
            pred_id = r[0] - word_offset
            if pred_id not in srl_rels:
                srl_rels[pred_id] = []
            srl_rels[pred_id].append((r[1] - word_offset, r[2] - word_offset, r[3]))
        samples.append((sentence, srl_rels, ner_spans))
        word_offset += len(sentence)
    return samples


def evaluate_retrieval(span_starts, span_ends, span_scores, pred_starts, pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
    """
      Evaluation for unlabeled retrieval.

      Args:
        gold_spans: Set of tuples of (start, end).
    """
    if len(span_starts) > 0:
        sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
            zip(span_starts, span_ends, span_scores),
            key=operator.itemgetter(2), reverse=True))
    else:
        sorted_starts = []
        sorted_ends = []
    for k, evaluator in evaluators.items():
        if k == -3:
            predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
        else:
            if k == -2:
                predicted_starts = pred_starts
                predicted_ends = pred_ends
                if debugging:
                    print("Predicted", zip(sorted_starts, sorted_ends, sorted_scores)[:len(gold_spans)])
                    print("Gold", gold_spans)
            # FIXME: scalar index error
            elif k == 0:
                is_predicted = span_scores > 0
                predicted_starts = span_starts[is_predicted]
                predicted_ends = span_ends[is_predicted]
            else:
                if k == -1:
                    num_predictions = len(gold_spans)
                else:
                    num_predictions = (k * text_length) / 100
                predicted_starts = sorted_starts[:num_predictions]
                predicted_ends = sorted_ends[:num_predictions]
            predicted_spans = set(zip(predicted_starts, predicted_ends))
        evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)


def _print_f1(total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
    return precision, recall, f1


class role_eval_metrix():
    def __init__(self, role):
        assert role in ["AGENT", "TARGET", "ALL"]
        self.role = role

        self.exact_match = 0
        self.exact_sys = 0
        self.exact_gold = 0
        self.exact_p = 0.0
        self.exact_r = 0.0
        self.exact_f = 0.0

        self.binary_match = 0
        self.binary_sys = 0
        self.binary_gold = 0
        self.binary_p = 0.0
        self.binary_r = 0.0
        self.binary_f = 0.0

        self.proportion_match = 0
        self.proportion_sys = 0
        self.proportion_gold = 0
        self.proportion_p = 0.0
        self.proportion_r = 0.0
        self.proportion_f = 0.0

    @staticmethod
    def comput_f1(match, sys, gold):
        try:
            p = 100.0 * match / sys
            r = 100.0 * match / gold
            f = 2 * p * r / (p + r)
            return p, r, f
        except Exception:
            print("Unable to compute P, R, F1 because of zero division", match, sys, gold)
            return 0.0, 0.0, 0.0

    def compute_exact_metric(self):
        print(self.role, "Exact number", self.exact_match, self.exact_sys, self.exact_gold)
        self.exact_p, self.exact_r, self.exact_f = role_eval_metrix.comput_f1(
            self.exact_match, self.exact_sys, self.exact_gold
        )
        print(self.role, "Exact Metric", self.exact_p, self.exact_r, self.exact_f)

    def compute_binary_metric(self):
        print(self.role, "Binary number", self.binary_match, self.binary_sys, self.binary_gold)
        self.binary_p, self.binary_r, self.binary_f = role_eval_metrix.comput_f1(
            self.binary_match, self.binary_sys, self.binary_gold
        )
        print(self.role, "Binary Metric", self.binary_p, self.binary_r, self.binary_f)

    def compute_proportion_metric(self):
        print(self.role, "Proportion number", self.proportion_match, self.proportion_sys, self.proportion_gold)
        self.proportion_p, self.proportion_r, self.proportion_f = role_eval_metrix.comput_f1(
            self.proportion_match, self.proportion_sys, self.proportion_gold
        )
        print(self.role, "Proportion Metric", self.proportion_p, self.proportion_r, self.proportion_f)

    def report_metric(self):
        self.compute_exact_metric()
        self.compute_binary_metric()
        self.compute_proportion_metric()

    def add_exact_matched(self, num):
        self.exact_match += num

    def add_exact_sys(self, num):
        self.exact_sys += num

    def add_exact_gold(self, num):
        self.exact_gold += num
        
    def add_binary_matched(self, num):
        self.binary_match += num

    def add_binary_sys(self, num):
        self.binary_sys += num

    def add_binary_gold(self, num):
        self.binary_gold += num

    def add_proportion_matched(self, num):
        self.proportion_match += num

    def add_proportion_sys(self, num):
        self.proportion_sys += num

    def add_proportion_gold(self, num):
        self.proportion_gold += num


def compute_orl_fl(gold_orl, dse_predictions, predictions, is_gold_dse=True):
    print('='*10, "Evaluating by the sys output and gold ans", '='*10)
    gold_dse, gold_spans, _ = gold_orl
    assert len(gold_spans) == len(predictions)
    # dse
    matched_dse_num, sys_dse_num, gold_dse_num = 0, 0, 0
    for gold_span, sys_span in zip(gold_dse, dse_predictions):
        gold_dse_num += len(gold_span)
        sys_dse_num += len(sys_span)
        for dse in sys_span:
            if dse in gold_span:
                matched_dse_num += 1
    #
    print("DSE info:", sys_dse_num, matched_dse_num, gold_dse_num)
    try:
        _print_f1(gold_dse_num, sys_dse_num, matched_dse_num, "DSE info")
    except Exception:
        print("DSE info:", sys_dse_num, matched_dse_num, gold_dse_num)
        pass
    #
    holder_metric, target_metic = role_eval_metrix("AGENT"), role_eval_metrix("TARGET")
    matched_span_num, sys_span_num, gold_span_num = 0, 0, 0
    matched_binary_num, sys_binary_num, gold_binary_num = 0, 0, 0
    matched_proportion_num, sys_proportion_num, gold_proportion_num = 0, 0, 0
    # check for all gold dse appers in sys dses
    if is_gold_dse is True:
        for gold_span, sys_span in zip(gold_spans, predictions):
            for key in gold_span:
                try:
                    assert key in sys_span
                except Exception:
                    print(key, "not in ", sys_span)
                    exit()

    def add_into_sys_and_gold(spans, mode="SYS"):
        if mode == "SYS":
            for sys_dse_arg in spans:
                sys_s, sys_e, label = sys_dse_arg
                if label == "AGENT":
                    holder_metric.add_exact_sys(1)
                    holder_metric.add_binary_sys(1)
                else:
                    assert label == "TARGET"
                    target_metic.add_exact_sys(1)
                    target_metic.add_binary_sys(1)
        else:
            assert mode == "GOLD"
            for sys_dse_arg in spans:
                sys_s, sys_e, label = sys_dse_arg
                if label == "AGENT":
                    holder_metric.add_exact_gold(1)
                    holder_metric.add_binary_gold(1)
                else:
                    assert label == "TARGET"
                    target_metic.add_exact_gold(1)
                    target_metic.add_binary_gold(1)

    hp_recall_matched, hp_recall_gold, hp_recall = 0, 0, 0.0
    hp_precision_mathced, hp_precision_gold, hp_precision = 0, 0, 0.0
    for gold_span, sys_span in zip(gold_spans, predictions):
        # print(gold_span, sys_span)
        if is_gold_dse is True:
            assert len(gold_span) == len(sys_span)
        # for the gold spans:
        for gold_dse in gold_span:
            gold_args = gold_span[gold_dse]
            gold_span_num += len(gold_args)  # add the gold number of total exact
            add_into_sys_and_gold(gold_args, "GOLD")  # add the gold number of agent/target binary and exact

            gold_binary_num += len(gold_args)  # add the gold number of total binary

            # gold span proportional num
            for gold_dse_arg in gold_args:
                gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                gold_proportion_num += 1
                if gold_arg_label == "ADR":
                    holder_metric.add_proportion_gold(1)
                else:
                    assert gold_arg_label == "ADR"
                    target_metic.add_proportion_gold(1)

        # for the sys spans
        for sys_dse in sorted(list(sys_span.keys())):
            if is_gold_dse is True:
                assert sys_dse in gold_span
            if sys_dse not in gold_span:
                assert is_gold_dse is False

            sys_args = sys_span[sys_dse]
            add_into_sys_and_gold(sys_args, "SYS")

            # exact F1 score process
            sys_span_num += len(sys_args)
            if sys_dse in gold_span:  # if the sys dse in the gold span, i.e., the sys dse is predicted correctly
                gold_args = gold_span[sys_dse]
                for sys_dse_arg in sys_args:
                    if sys_dse_arg in gold_args:
                        matched_span_num += 1
                        _, _, label = sys_dse_arg
                        if label == "ADR":
                            holder_metric.add_exact_matched(1)
                        else:
                            assert label == "ADR"
                            target_metic.add_exact_matched(1)
            # binary F1 score process
            sys_binary_num += len(sys_args)

            if sys_dse in gold_span:
                # labeled
                for sys_dse_arg in sys_args:
                    sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                    for gold_dse_arg in gold_args:
                        gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                        find = False
                        if sys_arg_label == gold_arg_label:  # if the label is the same
                            for i in range(sys_arg_start, sys_arg_end + 1):
                                if gold_arg_start <= i <= gold_arg_end:
                                    find = True
                                    if gold_arg_label == "ADR":
                                        holder_metric.add_binary_matched(1)
                                    else:
                                        assert gold_arg_label == "ADR"
                                        target_metic.add_binary_matched(1)
                                    break
                        if find is True:
                            matched_binary_num += 1
                            break
            # proportional num
            for sys_dse_arg in sys_args:
                sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                sys_proportion_num += 1
                if sys_arg_label == "ADR":
                    holder_metric.add_proportion_sys(1)
                else:
                    assert sys_arg_label == "ADR"
                    target_metic.add_proportion_sys(1)

                if sys_dse in gold_span:
                    # proportion
                    gold_args = gold_span[sys_dse]
                    proportions = []
                    for gold_dse_arg in gold_args:
                        gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                        gold_arg_length = gold_arg_end - gold_arg_start + 1
                        # gold_proportion_num += 1
                        count = 0
                        if gold_arg_label == sys_arg_label:
                            for i in range(gold_arg_start, gold_arg_end + 1):
                                if sys_arg_start <= i <= sys_arg_end:
                                    count += 1
                            proportions.append([count, gold_arg_label, gold_arg_length])
                    if len(proportions) > 0:
                        max_overlapped = sorted(proportions, reverse=True)[0]
                        max_overlapped_label = max_overlapped[1]
                        gold_arg_length = max_overlapped[2]
                        if max_overlapped_label == "ADR":
                            holder_metric.add_proportion_matched(1.0 * max_overlapped[0] / gold_arg_length)
                            matched_proportion_num += 1.0 * max_overlapped[0] / gold_arg_length
                            # print(1.0 * max_overlapped[0] / gold_arg_length)
                        else:
                            assert max_overlapped_label == "ADR"
                            target_metic.add_proportion_matched(1.0 * max_overlapped[0] / gold_arg_length)
                            matched_proportion_num += 1.0 * max_overlapped[0] / gold_arg_length
                            # print(1.0 * max_overlapped[0] / gold_arg_length)
                        # if max_overlapped[0] > 0:
                        #     print(sys_dse, 1.0 * max_overlapped[0] / gold_arg_length)

    print("Matched_span_num, sys_span_num, gold_span_num:", matched_span_num, sys_span_num, gold_span_num)
    print("Matched_binary_span_num, sys_binary_span_num, gold_binary_span_num:", \
          matched_binary_num, sys_binary_num, gold_binary_num)
    print("Matched_proportion_span_num, sys_proportion_span_num, gold_proportion_span_num:", \
          matched_proportion_num, sys_proportion_num, gold_proportion_num)
    try:
        p = 100.0 * matched_span_num / sys_span_num
        r = 100.0 * matched_span_num / gold_span_num
        f = 2.0 * p * r / (p + r)
        print("Look at this line =============================> ORL Exact (P, R, F1)", p, r, f)
        return f
    except Exception:
        return 0.0


def compute_orl_fl(gold_orl, dse_predictions, predictions, is_gold_dse=True):
    print('='*10, "Evaluating by the sys output and gold ans", '='*10)
    gold_dse, gold_spans, _ = gold_orl
    assert len(gold_spans) == len(predictions)
    # dse
    matched_dse_num, sys_dse_num, gold_dse_num = 0, 0, 0
    for gold_span, sys_span in zip(gold_dse, dse_predictions):
        gold_dse_num += len(gold_span)
        sys_dse_num += len(sys_span)
        for dse in sys_span:
            if dse in gold_span:
                matched_dse_num += 1
    #
    print("DSE info:", sys_dse_num, matched_dse_num, gold_dse_num)
    try:
        _print_f1(gold_dse_num, sys_dse_num, matched_dse_num, "DSE info")
    except Exception:
        print("DSE info:", sys_dse_num, matched_dse_num, gold_dse_num)
        pass
    #
    holder_metric, target_metic = role_eval_metrix("AGENT"), role_eval_metrix("TARGET")
    matched_span_num, sys_span_num, gold_span_num = 0, 0, 0
    matched_binary_num, sys_binary_num, gold_binary_num = 0, 0, 0
    matched_proportion_num, sys_proportion_num, gold_proportion_num = 0, 0, 0
    # check for all gold dse appers in sys dses
    if is_gold_dse is True:
        for gold_span, sys_span in zip(gold_spans, predictions):
            for key in gold_span:
                try:
                    assert key in sys_span
                except Exception:
                    print(key, "not in ", sys_span)
                    exit()

    def add_into_sys_and_gold(spans, mode="SYS"):
        if mode == "SYS":
            for sys_dse_arg in spans:
                sys_s, sys_e, label = sys_dse_arg
                if label == "AGENT":
                    holder_metric.add_exact_sys(1)
                    holder_metric.add_binary_sys(1)
                else:
                    assert label == "TARGET"
                    target_metic.add_exact_sys(1)
                    target_metic.add_binary_sys(1)
        else:
            assert mode == "GOLD"
            for sys_dse_arg in spans:
                sys_s, sys_e, label = sys_dse_arg
                if label == "AGENT":
                    holder_metric.add_exact_gold(1)
                    holder_metric.add_binary_gold(1)
                else:
                    assert label == "TARGET"
                    target_metic.add_exact_gold(1)
                    target_metic.add_binary_gold(1)

    hp_recall_matched, hp_recall_gold, hp_recall = 0, 0, 0.0
    hp_precision_mathced, hp_precision_gold, hp_precision = 0, 0, 0.0
    for gold_span, sys_span in zip(gold_spans, predictions):
        # print(gold_span, sys_span)
        if is_gold_dse is True:
            assert len(gold_span) == len(sys_span)
        # for the gold spans:
        for gold_dse in gold_span:
            gold_args = gold_span[gold_dse]
            gold_span_num += len(gold_args)  # add the gold number of total exact
            add_into_sys_and_gold(gold_args, "GOLD")  # add the gold number of agent/target binary and exact

            gold_binary_num += len(gold_args)  # add the gold number of total binary

            # gold span proportional num
            for gold_dse_arg in gold_args:
                gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                gold_proportion_num += 1
                if gold_arg_label == "AGENT":
                    holder_metric.add_proportion_gold(1)
                else:
                    assert gold_arg_label == "TARGET"
                    target_metic.add_proportion_gold(1)

        # for the sys spans
        for sys_dse in sorted(list(sys_span.keys())):
            if is_gold_dse is True:
                assert sys_dse in gold_span
            if sys_dse not in gold_span:
                assert is_gold_dse is False

            sys_args = sys_span[sys_dse]
            add_into_sys_and_gold(sys_args, "SYS")

            # exact F1 score process
            sys_span_num += len(sys_args)
            if sys_dse in gold_span:  # if the sys dse in the gold span, i.e., the sys dse is predicted correctly
                gold_args = gold_span[sys_dse]
                for sys_dse_arg in sys_args:
                    if sys_dse_arg in gold_args:
                        matched_span_num += 1
                        _, _, label = sys_dse_arg
                        if label == "AGENT":
                            holder_metric.add_exact_matched(1)
                        else:
                            assert label == "TARGET"
                            target_metic.add_exact_matched(1)
            # binary F1 score process
            sys_binary_num += len(sys_args)

            if sys_dse in gold_span:
                # labeled
                for sys_dse_arg in sys_args:
                    sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                    for gold_dse_arg in gold_args:
                        gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                        find = False
                        if sys_arg_label == gold_arg_label:  # if the label is the same
                            for i in range(sys_arg_start, sys_arg_end + 1):
                                if gold_arg_start <= i <= gold_arg_end:
                                    find = True
                                    if gold_arg_label == "AGENT":
                                        holder_metric.add_binary_matched(1)
                                    else:
                                        assert gold_arg_label == "TARGET"
                                        target_metic.add_binary_matched(1)
                                    break
                        if find is True:
                            matched_binary_num += 1
                            break
            # proportional num
            for sys_dse_arg in sys_args:
                sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                sys_proportion_num += 1
                if sys_arg_label == "AGENT":
                    holder_metric.add_proportion_sys(1)
                else:
                    assert sys_arg_label == "TARGET"
                    target_metic.add_proportion_sys(1)

                if sys_dse in gold_span:
                    # proportion
                    gold_args = gold_span[sys_dse]
                    proportions = []
                    for gold_dse_arg in gold_args:
                        gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                        gold_arg_length = gold_arg_end - gold_arg_start + 1
                        # gold_proportion_num += 1
                        count = 0
                        if gold_arg_label == sys_arg_label:
                            for i in range(gold_arg_start, gold_arg_end + 1):
                                if sys_arg_start <= i <= sys_arg_end:
                                    count += 1
                            proportions.append([count, gold_arg_label, gold_arg_length])
                    if len(proportions) > 0:
                        max_overlapped = sorted(proportions, reverse=True)[0]
                        max_overlapped_label = max_overlapped[1]
                        gold_arg_length = max_overlapped[2]
                        if max_overlapped_label == "AGENT":
                            holder_metric.add_proportion_matched(1.0 * max_overlapped[0] / gold_arg_length)
                            matched_proportion_num += 1.0 * max_overlapped[0] / gold_arg_length
                            # print(1.0 * max_overlapped[0] / gold_arg_length)
                        else:
                            assert max_overlapped_label == "TARGET"
                            target_metic.add_proportion_matched(1.0 * max_overlapped[0] / gold_arg_length)
                            matched_proportion_num += 1.0 * max_overlapped[0] / gold_arg_length
                            # print(1.0 * max_overlapped[0] / gold_arg_length)
                        # if max_overlapped[0] > 0:
                        #     print(sys_dse, 1.0 * max_overlapped[0] / gold_arg_length)

    print("Matched_span_num, sys_span_num, gold_span_num:", matched_span_num, sys_span_num, gold_span_num)
    print("Matched_binary_span_num, sys_binary_span_num, gold_binary_span_num:", \
          matched_binary_num, sys_binary_num, gold_binary_num)
    print("Matched_proportion_span_num, sys_proportion_span_num, gold_proportion_span_num:", \
          matched_proportion_num, sys_proportion_num, gold_proportion_num)
    # hp_recall = 100.0 * hp_recall_matched / hp_recall_gold
    # hp_precision = 100.0 * hp_precision_mathced / hp_precision_gold
    # print(hp_precision_mathced, hp_precision_gold, hp_recall_matched, hp_recall_gold)
    # print("Holder", hp_precision, hp_recall, 2 * hp_precision * hp_recall / (hp_precision + hp_recall))
    # print("The averaged number of sys args is {}".format(1.0 * sys_proportion_num / sys_span_num))

    try:
        # eval_spans_by_length_and_range(gold_orl, predictions)
        # analyze_spans(gold_orl, predictions)
        pass
    except Exception:
        print("something happens in eval spans by length and range!")
    try:
        p = 100.0 * matched_span_num / sys_span_num
        r = 100.0 * matched_span_num / gold_span_num
        f = 2.0 * p * r / (p + r)
        print("ORL Exact (P, R, F1)", p, r, f)
        bp = 100.0 * matched_binary_num / sys_binary_num
        br = 100.0 * matched_binary_num / gold_binary_num
        bf = 2.0 * bp * br / (bp + br)
        print("ORL Binary (P, R, F1)", bp, br, bf)
        pp = 100.0 * matched_proportion_num / sys_proportion_num
        pr = 100.0 * matched_proportion_num / gold_proportion_num
        pf = 2.0 * pp * pr / (pp + pr)
        print("ORL Proportion (P, R, F1)", pp, pr, pf)
        holder_metric.report_metric()
        target_metic.report_metric()
        return f
    except Exception:
        return 0.0


class eval_by_length_and_range():
    def __init__(self):
        self.span_length = [10, 20, 30, 40]
        self.span_range = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 4}
        self.max_range_threshold = max(self.span_range.keys())
        self.range_map = {0: '0-1', 1: '2-3', 2: '4-7', 3: '8-11', 4: '12-MAX'}
        self.eval_span_length = [role_eval_metrix("ALL") for _ in range(len(self.span_length) + 1)]
        self.eval_span_range = [role_eval_metrix("ALL") for _ in range(len(set(self.span_range.values())))]
        print("eval info by range len(self.eval_span_range)", len(self.eval_span_range))

    def return_span_length(self, span):
        start, end, label = span
        len = end - start + 1
        return len

    def add_by_length_gold(self, span):
        span_len = self.return_span_length(span)
        idx = int(span_len / 10)
        self.eval_span_length[idx].add_exact_gold(1)
        self.eval_span_length[idx].add_binary_gold(1)

    def add_by_length_sys(self, span):
        span_len = self.return_span_length(span)
        idx = int(span_len / 10)
        self.eval_span_length[idx].add_exact_sys(1)
        self.eval_span_length[idx].add_binary_sys(1)

    def add_by_length_matched(self, span):
        span_len = self.return_span_length(span)
        idx = int(span_len / 10)
        self.eval_span_length[idx].add_exact_matched(1)

    def add_by_length_binary_matched(self, span):
        span_len = self.return_span_length(span)
        idx = int(span_len / 10)
        self.eval_span_length[idx].add_binary_matched(1)

    def compute_eval_by_length(self):
        for length, item in zip([0] + self.span_length, self.eval_span_length):
            print("Exact F1 by span length", str(length))
            item.compute_exact_metric()

    def compute_binary_eval_by_length(self):
        for length, item in zip([0] + self.span_length, self.eval_span_length):
            print("Binary F1 by span length", str(length))
            item.compute_binary_metric()

    def return_span_range(self, span, dse):
        span_start, span_end, span_label = span
        expression_start, expression_end = dse
        if span_end < expression_start:  # conditional one
            range_from_expression = expression_start - span_end
            if range_from_expression > self.max_range_threshold:
                range_from_expression = self.max_range_threshold
        else:  # conditin two
            assert expression_end < span_start
            range_from_expression = span_start - expression_end
            if range_from_expression > self.max_range_threshold:
                range_from_expression = self.max_range_threshold
        return range_from_expression

    def add_by_range_gold(self, span, dse):
        span_range_idx = self.return_span_range(span, dse)
        # print("gold", span_range_idx)
        span_range_idx = self.span_range[span_range_idx]
        self.eval_span_range[span_range_idx].add_exact_gold(1)
        self.eval_span_range[span_range_idx].add_binary_gold(1)
        # print("gold", span_range_idx, "done.")

    def add_by_range_sys(self, span, dse):
        span_range_idx = self.return_span_range(span, dse)
        # print("sys", span_range_idx)
        span_range_idx = self.span_range[span_range_idx]
        self.eval_span_range[span_range_idx].add_exact_sys(1)
        self.eval_span_range[span_range_idx].add_binary_sys(1)
        # print("sys", span_range_idx, "done.")

    def add_by_range_matched(self, span, dse):
        span_range_idx = self.return_span_range(span, dse)
        # print("matched", span_range_idx)
        span_range_idx = self.span_range[span_range_idx]
        self.eval_span_range[span_range_idx].add_exact_matched(1)
        # print("matched", span_range_idx, "done")

    def add_by_range_binary_matched(self, span, dse):
        span_range_idx = self.return_span_range(span, dse)
        # print("matched", span_range_idx)
        span_range_idx = self.span_range[span_range_idx]
        self.eval_span_range[span_range_idx].add_binary_matched(1)

    def compute_eval_by_range(self):
        for i, item in enumerate(self.eval_span_range):
            print("Exact F1 by range length", self.range_map[i])
            item.compute_exact_metric()

    def compute_binary_eval_by_range(self):
        for i, item in enumerate(self.eval_span_range):
            print("Binary F1 by range length", self.range_map[i])
            item.compute_binary_metric()


def eval_spans_by_length_and_range(gold_orl, predictions):
    gold_dse, gold_spans, _ = gold_orl
    assert len(gold_spans) == len(predictions)

    eval_info = eval_by_length_and_range()
    eval_range_info = eval_by_length_and_range()

    def count_gold_by_length(spans, mode="GOLD"):
        if mode == "GOLD":
            for span in spans:
                eval_info.add_by_length_gold(span)
        else:
            assert mode == "SYS"
            for span in spans:
                eval_info.add_by_length_sys(span)

    def count_gold_by_range(spans, dse, mode="GOLD"):
        if mode == "GOLD":
            for span in spans:
                eval_range_info.add_by_range_gold(span, dse)
        else:
            assert mode == "SYS"
            for span in spans:
                eval_range_info.add_by_range_sys(span, dse)

    for gold_span, sys_span in zip(gold_spans, predictions):
        for sys_dse in sys_span:
            if gold_dse is True:
                assert sys_dse in gold_span
            if sys_dse not in gold_span:
                print(sys_dse, "not in gold dse", gold_span.keys(),\
                      "please check whether the dse is gold!, currently not supported sys dse")
                exit()
            sys_args = sys_span[sys_dse]
            gold_args = gold_span[sys_dse]

            count_gold_by_length(sys_args, "SYS")
            count_gold_by_length(gold_args, "GOLD")

            count_gold_by_range(sys_args, sys_dse, "SYS")
            count_gold_by_range(gold_args, sys_dse, "GOLD")

            for sys_dse_arg in sys_args:
                if sys_dse_arg in gold_args:
                    _, _, label = sys_dse_arg
                    eval_info.add_by_length_matched(sys_dse_arg)
                    eval_range_info.add_by_range_matched(sys_dse_arg, sys_dse)
                    if label == "AGENT":
                        pass
                    else:
                        assert label == "TARGET"
                        pass
            # labeled
            for sys_dse_arg in sys_args:
                sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                for gold_dse_arg in gold_args:
                    gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                    find = False
                    if sys_arg_label == gold_arg_label:  # if the label is the same
                        for i in range(sys_arg_start, sys_arg_end + 1):
                            if gold_arg_start <= i <= gold_arg_end:
                                find = True
                                eval_info.add_by_length_binary_matched(sys_dse_arg)
                                eval_range_info.add_by_range_binary_matched(sys_dse_arg, sys_dse)
                                if gold_arg_label == "AGENT":
                                    # holder_metric.add_binary_matched(1)
                                    pass
                                else:
                                    # assert gold_arg_label == "TARGET"
                                    # target_metic.add_binary_matched(1)
                                    pass
                                break
                    if find is True:
                        # matched_binary_num += 1
                        break
    eval_info.compute_eval_by_length()
    eval_range_info.compute_eval_by_range()

    eval_info.compute_binary_eval_by_length()
    eval_range_info.compute_binary_eval_by_range()


def analyze_spans(gold_orl, predictions):
    gold_dse, gold_spans, _ = gold_orl
    assert len(gold_spans) == len(predictions)

    exceed_num, miss_num, span_error_num, label_error = 0, 0, 0, 0
    predict_correct = 0
    gold_predict_correct, gold_miss_num, gold_span_error_num , gold_label_error = 0, 0, 0, 0
    gold_binary_matched = 0
    set_sys_span_error, set_gold_span_error = set(), set()
    for gold_span, sys_span in zip(gold_spans, predictions):
        for sys_dse in sys_span:
            if gold_dse is True:
                assert sys_dse in gold_span
            if sys_dse not in gold_span:
                print(sys_dse, "not in gold dse", gold_span.keys(),\
                      "please check whether the dse is gold!, currently not supported sys dse")
                exit()
            sys_args = sys_span[sys_dse]
            gold_args = gold_span[sys_dse]

            # exceed predict
            for sys_dse_arg in sys_args:
                sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                flag = False
                counts = [(0, "O", -1)]
                for j, gold_dse_arg in enumerate(gold_args):
                    gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                    count = 0
                    for i in range(sys_arg_start, sys_arg_end + 1):
                        if gold_arg_start <= i <= gold_arg_end:  # TODO, cannot ensure sys_arg == gold_arg
                            flag = True
                            count += 1
                    counts.append((count, gold_arg_label, j))
                counts = sorted(counts, reverse=True)[0]
                if flag is False:
                    exceed_num += 1
                else:
                    overlapped_gold_span = gold_args[counts[2]]
                    gold_start, gold_end, gold_span_label = overlapped_gold_span
                    if gold_start == sys_arg_start and gold_end == sys_arg_end:
                        if gold_span_label == sys_arg_label:
                            predict_correct += 1
                        else:
                            label_error += 1
                    else:
                        span_error_num += 1
                        set_sys_span_error.add(sys_dse_arg)

            # miss predict
            for gold_dse_arg in gold_args:
                gold_arg_start, gold_arg_end, gold_arg_label = gold_dse_arg
                flag = False
                counts = [(0, "O", -1)]
                for j, sys_dse_arg in enumerate(sys_args):
                    sys_arg_start, sys_arg_end, sys_arg_label = sys_dse_arg
                    count = 0
                    for i in range(gold_arg_start, gold_arg_end + 1):
                        if sys_arg_start <= i <= sys_arg_end:
                            count += 1
                            flag = True
                    counts.append((count, sys_arg_label, j))
                counts = sorted(counts, reverse=True)[0]
                if flag is False:
                    miss_num += 1
                else:
                    gold_binary_matched += 1
                    overlapped_sys_span = sys_args[counts[2]]
                    sys_start, sys_end, sys_span_label = overlapped_sys_span
                    if gold_arg_start == sys_start and gold_arg_end == sys_end:
                        if gold_arg_label == sys_span_label:
                            gold_predict_correct += 1
                        else:
                            gold_label_error += 1
                    else:
                        gold_span_error_num += 1
                        set_gold_span_error.add(gold_dse_arg)
    print("="*10, "Error Analysis", "="*10)
    print("Exceed error num", exceed_num, "Miss error num", miss_num, "Span error num", span_error_num,\
          "label_error", label_error)
    print("predict_correct", predict_correct, "gold_binary_matched", gold_binary_matched)
    print("gold_predict_correct", gold_predict_correct, "gold_span_error_num", gold_span_error_num,\
          "gold_label_error", gold_label_error)


class eval_output_sentence():
    def __init__(self, tokens):
        self.words = tokens[1]
        self.sentence = " ".join(self.words)
        self.gold_label = tokens[2]
        self.sys_label = ["O"] * len(self.gold_label)
        self.dse = None
        self.dse_label = []
        self.extract_dse()

    def extract_dse(self):
        dse_s, dse_e = -1, -1
        for i, label in enumerate(self.gold_label):
            if label.startswith("S-DSE"):
                dse_s = dse_e = i
                self.dse_label.append(label)
            elif label.startswith("B-DSE"):
                dse_s = i
                self.dse_label.append(label)
            elif label.startswith("M-DSE"):
                self.dse_label.append(label)
            elif label.startswith("E-DSE"):
                dse_e = i
                self.dse_label.append(label)
            else:
                pass
        assert dse_s != -1 and dse_e != -1
        self.dse = (dse_s, dse_e)
        # print(self.dse_label, self.dse)
        assert len(self.dse_label) == dse_e - dse_s + 1


class EvalOverlapConsOrlArguments():
    def __init__(self):
        self.all_matched = 0
        self.all_gold = 0
        self.agent_matched = 0
        self.agent_gold = 0
        self.target_matched = 0
        self.target_gold = 0
        self.info = ""

        self.all_matched_cons_label = Counter()
        self.agent_matched_cons_label = Counter()
        self.target_matched_cons_label = Counter()

        self.all_matched_argu_cons_label = Counter()
        self.agent_matched_argu_cons_label = Counter()
        self.target_matched_argu_cons_label = Counter()

    def compute_list_of_spans(self, orl_spans, cons_spans, argu_cons=None):
        for orl_span in orl_spans:
            self.compute(orl_span, cons_spans, argu_cons)

    def compute(self, orl_span, cons_spans, argu_cons=None):
        dse_start, dse_end, arg_start, arg_end, orl_label = orl_span
        self.all_gold += 1
        if orl_label == "AGENT":
            self.agent_gold += 1
        else:
            assert orl_label == "TARGET"
            self.target_gold += 1
        for cons_span in cons_spans:
            cons_start, cons_end, cons_label = cons_span
            orl_span = (arg_start, arg_end)
            cons_span = (cons_start, cons_end)
            if orl_span == cons_span:
                self.all_matched += 1
                self.all_matched_cons_label[cons_label] += 1
                if orl_label == "AGENT":
                    self.agent_matched += 1
                    self.agent_matched_cons_label[cons_label] += 1
                else:
                    assert orl_label == "TARGET"
                    self.target_matched += 1
                    self.target_matched_cons_label[cons_label] += 1
                break
            else:
                pass
        # ORL predicted arguments' constituents
        if argu_cons is not None:
            for cons_span in argu_cons:
                cons_start, cons_end, cons_label = cons_span
                self.all_matched_argu_cons_label[cons_label] += 1

    def output(self):
        try:
            print("="*10, "info on the constituent spans overlap the orl spans", "=" * 10)
            print("ALL (Recall)", 1.0 * self.all_matched / self.all_gold)
            print("AGENT (Recall)", 1.0 * self.agent_matched / self.agent_gold)
            print("TARGET (Recall)", 1.0 * self.target_matched / self.target_gold)
            print("ALL matched cons labels", self.all_matched_cons_label)
            print("AGENT matched cons labels", self.agent_matched_cons_label)
            print("TARGET matched cons labels", self.target_matched_cons_label)
            print("Predicted arguments' corresponding predicted cons labels", self.all_matched_argu_cons_label)
        except Exception:
            print("division by zero!")


def output_predicted_filex(gold_orl, predictions, outputpath=None):
    _, _, sentences = gold_orl
    sys_data = OrderedDict()
    for sentence, sys_span in zip(sentences, predictions):
        # assert sentence not in sys_data
        sys_data[sentence] = sys_span
    print("Load {} unique test samples.".format(len(sys_data)))

    _, ans, sentences = gold_orl
    output_file = open(outputpath, 'w')
    for idx in range(len(sentences)):
        # print(sample)
        sentence = sentences[idx]
        gold_ans = ans[idx]
        gold_orl = []
        for key in gold_ans:
            dse_s, dse_e = key
            gold_orl.append([dse_s, dse_e, dse_s, dse_e, "DSE"])
            for orl_span in gold_ans[key]:
                span_s, span_e, span_label = orl_span
                gold_orl.append([dse_s, dse_e, span_s, span_e, span_label])
        sys_ans = predictions[idx]
        sys_orl = []
        for key in sys_ans:
            dse_s, dse_e = key
            for orl_span in sys_ans[key]:
                span_s, span_e, span_label = orl_span
                sys_orl.append([int(dse_s), int(dse_e), int(span_s), int(span_e), span_label])
        #
        output = {
            "sentence": [sentence.split()],
            "gold_orl": gold_orl,
            "sys_orl": sys_orl,
        }
        output_file.write(json.dumps(output) + '\n')
    print("Output file ".format(output_file))
    output_file.close()


def output_predicted_file(eval_file, gold_orl, predictions, sys_args, outputpath=None, sys_cons=None):
    with open(eval_file, 'r') as input_orl_file:
        dataset = []
        sentence = []
        for line in input_orl_file.readlines():
            if line.strip() == "":
                tuples = list(zip(*sentence))
                dataset.append(eval_output_sentence(tuples))
                sentence = []
                continue
            tokens = line.strip().split()
            # print(tokens)
            sentence.append(tokens)
        print("Load {} test samples for gold file {}".format(len(dataset), eval_file))

    _, _, sentences = gold_orl
    sys_data = OrderedDict()
    for sentence, sys_span in zip(sentences, predictions):
        # assert sentence not in sys_data
        sys_data[sentence] = sys_span
    print("Load {} unique test samples.".format(len(sys_data)))

    for sample in dataset:
        sentence = sample.sentence
        assert sentence in sys_data.keys()
        sys_predictions = sys_data[sentence]
        # print(sample.sentence, sample.dse, sys_predictions.keys())
        # add gold dse
        for idx, dse_label in zip(range(sample.dse[0], sample.dse[1] + 1), sample.dse_label):
            assert sample.sys_label[idx] == "O"
            sample.sys_label[idx] = dse_label

        if sample.dse in sys_predictions.keys():
            assert sample.dse in list(sys_predictions.keys())
            orl_spans = sys_predictions[sample.dse]
            # add sys argument
            for span in orl_spans:
                span_start, span_end, span_label = span
                if span_start == span_end:
                    assert sample.sys_label[span_start] == "O"
                    sample.sys_label[span_start] = "S-" + span_label
                else:
                    assert sample.sys_label[span_start] == "O"
                    assert sample.sys_label[span_end] == "O"
                    sample.sys_label[span_start] = "B-" + span_label
                    sample.sys_label[span_end] = "E-" + span_label
                    for internal in range(span_start + 1, span_end):
                        assert sample.sys_label[internal] == "O"
                        sample.sys_label[internal] = "M-" + span_label

    outputpath = eval_file + ".out" if outputpath is None else outputpath
    with open(outputpath, 'w') as output_file:
        for sample in dataset:
            # print(sample)
            sentence = sample.words
            gold_label = sample.gold_label
            sys_label = sample.sys_label
            xxx = zip(sentence, gold_label, sys_label)
            for i, info in enumerate(xxx):
                output_file.write(str(i) + '\t' + '\t'.join(info) + '\n')
            output_file.write('\n')
        print("output file {}".format(outputpath))

    if sys_cons is not None:
        sys_cons, sys_argu_cons = sys_cons
        _, ans, sentences = gold_orl
        cons_outputpath = eval_file + ".w.cons.out" if outputpath is None else outputpath + ".cons.json"
        output_file = open(cons_outputpath, 'w')
        eval_cons = EvalOverlapConsOrlArguments()
        for idx in range(len(sentences)):
            # print(sample)
            sentence = sentences[idx]
            gold_ans = ans[idx]
            gold_orl = []
            for key in gold_ans:
                dse_s, dse_e = key
                gold_orl.append([dse_s, dse_e, dse_s, dse_e, "DSE"])
                for orl_span in gold_ans[key]:
                    span_s, span_e, span_label = orl_span
                    gold_orl.append([dse_s, dse_e, span_s, span_e, span_label])
            sys_ans = predictions[idx]
            sys_orl = []
            for key in sys_ans:
                dse_s, dse_e = key
                for orl_span in sys_ans[key]:
                    span_s, span_e, span_label = orl_span
                    sys_orl.append([int(dse_s), int(dse_e), int(span_s), int(span_e), span_label])
            # sys arg
            sys_argument = sys_args[idx]
            sys_argument = sorted(sys_argument)
            #
            output = {
                "sentence": [sentence.split()],
                "gold_orl": gold_orl,
                "sys_orl": sys_orl,
                "sys_argument": sys_argument,
            }
            if len(sys_argu_cons) == 0:
                pass
            else:
                argu_cons = sys_argu_cons[idx]
                argu_cons = sorted(argu_cons, reverse=False)
                cons = sys_cons[idx]
                cons = sorted(cons, reverse=False)
                output.update(
                    {
                        "sys_argus_constituents": argu_cons,
                        "constituent": cons
                    }
                )
                eval_cons.compute_list_of_spans(gold_orl, cons, argu_cons)
            output_file.write(json.dumps(output) + '\n')
        print("Output file with constituent spans {}".format(cons_outputpath))
        eval_cons.output()
        output_file.close()


def compute_span_f1(gold_data, predictions, task_name):
    assert len(gold_data) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    label_confusions = Counter()  # Counter of (gold, pred) label pairs.

    for i in range(len(gold_data)):
        gold = gold_data[i]
        pred = predictions[i]
        total_gold += len(gold)
        total_predicted += len(pred)
        for a0 in gold:
            for a1 in pred:
                if a0[0] == a1[0] and a0[1] == a1[1]:
                    total_unlabeled_matched += 1
                    label_confusions.update([(a0[2], a1[2]), ])
                    if a0[2] == a1[2]:
                        total_matched += 1
    prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled " + task_name)
    return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_unlabeled_span_f1(gold_data, predictions, task_name):
    assert len(gold_data) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    label_confusions = Counter()  # Counter of (gold, pred) label pairs.

    for i in range(len(gold_data)):
        gold = gold_data[i]
        pred = predictions[i]
        total_gold += len(gold)
        total_predicted += len(pred)
        for a0 in gold:
            for a1 in pred:
                if a0[0] == a1[0] and a0[1] == a1[1]:
                    total_unlabeled_matched += 1
                    label_confusions.update([(a0[2], a1[2]), ])
                    if a0[2] == a1[2]:
                        total_matched += 1
    prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled " + task_name)
    return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_srl_f1(sentences, gold_srl, predictions, srl_conll_eval_path, span_based=True, auto_senses=""):
    """
    For dependency-based SRL (CoNLL-2009), the senses of the predicate are predicted offline, and the model didn't process it,
        so it is needed to provide the auto_sense file.
    """
    assert len(gold_srl) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    comp_sents = 0
    label_confusions = Counter()

    # Compute unofficial F1 of SRL relations.
    for gold, prediction in zip(gold_srl, predictions):
        gold_rels = 0
        pred_rels = 0
        matched = 0
        for pred_id, gold_args in gold.items():
            filtered_gold_args = [a for a in gold_args if a[2] not in ["V", "C-V"]]
            total_gold += len(filtered_gold_args)
            gold_rels += len(filtered_gold_args)
            if pred_id not in prediction:
                continue
            for a0 in filtered_gold_args:
                for a1 in prediction[pred_id]:
                    if a0[0] == a1[0] and a0[1] == a1[1]:
                        total_unlabeled_matched += 1
                        label_confusions.update([(a0[2], a1[2]), ])
                        if a0[2] == a1[2]:
                            total_matched += 1
                            matched += 1
        for pred_id, args in prediction.items():
            filtered_args = [a for a in args if a[2] not in ["V"]]  # "C-V"]]
            total_predicted += len(filtered_args)
            pred_rels += len(filtered_args)

        if gold_rels == matched and pred_rels == matched:
            comp_sents += 1

    precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "SRL (unofficial)")
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled SRL (unofficial)")

    # Prepare to compute official F1.
    if not srl_conll_eval_path:
        print("No gold conll_eval data provided. Recreating ...")
        gold_path = "/tmp/srl_pred_%d.gold" % os.getpid()
        print_to_conll(sentences, gold_srl, gold_path, None)
        gold_predicates = None
    else:
        word_based = span_based is not True
        # print(span_based, word_based)
        gold_path = srl_conll_eval_path
        gold_predicates = read_gold_predicates(gold_path, conll09=word_based)

    temp_output = "/tmp/srl_pred_%d.tmp" % os.getpid()
    print("Output temp outoput {}".format(temp_output))
    if word_based is True:
        print_to_conll(sentences, predictions, temp_output, gold_predicates, gold_path=gold_path, auto_senses=auto_senses)
    else:
        print_to_conll_span(sentences, predictions, temp_output, gold_predicates)

    # Evalute twice with official script.
    _SRL_EVAL_SCRIPT = _SRL_CONLL09_EVAL_SCRIPT if word_based else _SRL_CONLL_EVAL_SCRIPT
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_EVAL_SCRIPT, gold_path, temp_output), shell=True, stdout=subprocess.PIPE)
    eval_info = child.communicate()[0]
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_EVAL_SCRIPT, temp_output, gold_path), shell=True, stdout=subprocess.PIPE)
    eval_info2 = child2.communicate()[0]
    try:
        if word_based:  # word-based SRL
            for line in eval_info.split('\n'):
                if "Labeled recall" in line:
                    conll_recall = float(line.strip().split()[-2])
            for line in eval_info2.split('\n'):
                if "Labeled recall" in line:
                    conll_precision = float(line.strip().split()[-2])
            if conll_recall + conll_precision > 0:
                conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
            else:
                conll_f1 = 0
        else:
            eval_info = eval_info.decode()
            eval_info2 = eval_info2.decode()
            conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
            conll_precision = float(eval_info2.strip().split("\n")[6].strip().split()[5])
            if conll_recall + conll_precision > 0:
                conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
            else:
                conll_f1 = 0
        print(eval_info)
        print(eval_info2)
        print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
            conll_precision, conll_recall, conll_f1))
    except IndexError:
        conll_recall = 0
        conll_precision = 0
        conll_f1 = 0
        print("Unable to get FScore. Skipping.")
    return precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, label_confusions, comp_sents


def print_sentence_to_conll(fout, tokens, labels, pre_tokens=None):
    """Print a labeled sentence into CoNLL format.
  """
    for label_column in labels:
        assert len(label_column) == len(tokens)
    for i in range(len(tokens)):
        if pre_tokens is not None:
            fout.write(pre_tokens[i] + '\t')
            srl_labels = []
            for label_column in labels:
                if label_column[i] == 'V':
                    srl_labels.append('_')
                else:
                    srl_labels.append(label_column[i])
            fout.write('\t'.join(srl_labels))
        else:
            fout.write(tokens[i].ljust(15))
            for label_column in labels:
                fout.write(label_column[i].rjust(15))
        fout.write("\n")
    fout.write("\n")


def read_gold_predicates(gold_path, conll09=False):
    print("gold path", gold_path)
    fin = codecs.open(gold_path, "r", "utf-8")
    gold_predicates = [[], ]
    for line in fin:
        line = line.strip()
        if not line:
            gold_predicates.append([])
        else:
            if conll09 is True:
                info = line.split("\t")
                gold_predicates[-1].append(info[13])
            else:
                info = line.split()
                gold_predicates[-1].append(info[0])
    fin.close()
    return gold_predicates


def get_pre_tokens(file_path):
    gold_ans = codecs.open(file_path, 'r', "utf-8")
    gold_sentences, sen = [], []
    for line in gold_ans.readlines():
        if line == '\n' or line == '\r\n' or line.strip() == '':
            gold_sentences.append(sen)
            sen = []
            continue
        pre_tokens = line.split()[:14]
        pre_tokens = '\t'.join(pre_tokens)
        sen.append(pre_tokens)
    return gold_sentences


def print_to_conll(sentences, srl_labels, output_filename, gold_predicates, gold_path=None, auto_senses=None):
    gold_sentences = None
    if auto_senses != "":
        gold_sentences = get_pre_tokens(auto_senses)
    elif gold_path is not None:
        gold_sentences = get_pre_tokens(gold_path)
    else:
        print("No gold path or auto senses provided!")
        exit()
    fout = codecs.open(output_filename, "w", "utf-8")
    for sent_id, words in enumerate(sentences):
        if gold_predicates:
            assert len(gold_predicates[sent_id]) == len(words)
        pred_to_args = srl_labels[sent_id]
        props = ["-" for _ in words]
        if gold_path is not None:
            col_labels = [["_" for _ in words] for _ in range(len(pred_to_args))]
        else:
            col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
        for i, pred_id in enumerate(sorted(pred_to_args.keys())):
            # To make sure CoNLL-eval script count matching predicates as correct.
            if gold_predicates and gold_predicates[sent_id][pred_id] != "-":
                props[pred_id] = gold_predicates[sent_id][pred_id]
            else:
                props[pred_id] = "P" + words[pred_id]
            flags = [False for _ in words]
            for start, end, label in pred_to_args[pred_id]:
                if gold_path is not None:
                    if not max(flags[start:end + 1]):
                        col_labels[i][start] = label
                        col_labels[i][end] = col_labels[i][end]
                        for j in range(start, end + 1):
                            flags[j] = True
                else:
                    if not max(flags[start:end + 1]):
                        col_labels[i][start] = "(" + label + col_labels[i][start]
                        col_labels[i][end] = col_labels[i][end] + ")"
                        for j in range(start, end + 1):
                            flags[j] = True
            # Add unpredicted verb (for predicted SRL).
            if not flags[pred_id]:  # if the predicate id is False
                col_labels[i][pred_id] = "(V*)"
        print_sentence_to_conll(fout, props, col_labels, pre_tokens=gold_sentences[sent_id])
    fout.close()


def print_to_conll_span(sentences, srl_labels, output_filename, gold_predicates):
    fout = codecs.open(output_filename, "w", "utf-8")
    for sent_id, words in enumerate(sentences):
        if gold_predicates:
            assert len(gold_predicates[sent_id]) == len(words)
        pred_to_args = srl_labels[sent_id]
        props = ["-" for _ in words]
        col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
        for i, pred_id in enumerate(sorted(pred_to_args.keys())):
            # To make sure CoNLL-eval script count matching predicates as correct.
            if gold_predicates and gold_predicates[sent_id][pred_id] != "-":
                props[pred_id] = gold_predicates[sent_id][pred_id]
            else:
                props[pred_id] = "P" + words[pred_id]
            flags = [False for _ in words]
            for start, end, label in pred_to_args[pred_id]:
                if not max(flags[start:end + 1]):
                    col_labels[i][start] = "(" + label + col_labels[i][start]
                    col_labels[i][end] = col_labels[i][end] + ")"
                    for j in range(start, end + 1):
                        flags[j] = True
            # Add unpredicted verb (for predicted SRL).
            if not flags[pred_id]:  # if the predicate id is False
                col_labels[i][pred_id] = "(V*)"
        # print(gold_predicates[sent_id])
        # print(pred_to_args)
        # print(props, col_labels)
        # exit()
        print_sentence_to_conll(fout, props, col_labels)
    fout.close()
