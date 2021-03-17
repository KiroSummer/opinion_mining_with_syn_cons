import json
import sys

from collections import OrderedDict, Counter


DSE="DSE"
AGENT="AGENT"
TARGET="TARGET"


class Sample():
    def __init__(self, obj):
        self.sentence = obj["sentence"]
        self.gold_orl = obj['gold_orl']
        self.sys_orl = obj["sys_orl"]
        # self.sys_argus_constituents = obj['sys_argus_constituents']
        # self.constituent = obj["constituent"]


class ConllSample():
    def __init__(self, tokens):
        info = list(zip(*tokens))
        self.sentence = info[0]
        self.gold_orl_tokens = info[1]
        self.sys_orl_tokens = info[2]
        # print(self.sentence)
        # print(self.gold_orl)
        # print(self.sys_orl)
        self.gold_orl = []
        self.sys_orl = []
        self.generate_orl_relations()

    def generate_tuples(self, tokens):
        orl_tuples = []
        orl_tuple = []
        dse = []
        for i, t in enumerate(tokens):
            if DSE in t:  # dse don't participate in evaluation
                if t.startswith("B"):
                    dse.append(i)
                elif t.startswith("M"):
                    continue
                elif t.startswith("E"):
                    dse.append(i)
                elif t.startswith("S"):
                    dse = [i, i]
                continue
            if t.startswith("B"):
                orl_tuple.append(i)
            elif t.startswith("M"):
                continue
            elif t.startswith("E"):
                orl_tuple.append(i)
                orl_tuple.append(t[2:])

                orl_tuples.append(orl_tuple)
                orl_tuple = []
            elif t.startswith("S"):
                orl_tuple = [i, i, t[2:]]
                orl_tuples.append(orl_tuple)
                orl_tuple = []
            else:
                assert t == "O"
        assert len(dse) == 2
        result = []
        for t in orl_tuples:
            result.append(dse + t)
        return result

    def generate_orl_relations(self):
        self.gold_orl = self.generate_tuples(self.gold_orl_tokens)
        self.sys_orl = self.generate_tuples(self.sys_orl_tokens)
        # print(self.gold_orl, self.sys_orl)


def load_eval_data(eval_path):
    eval_data = []
    sample = []
    with open(eval_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":  # end of sample
                eval_data.append(ConllSample(sample))
                # exit()
                sample = []
                continue
            tokens = line.split()
            sample.append(tokens[1:])
    print("Loaded {} eval examples.".format(len(eval_data)))
    return eval_data


class CountMetric(object):
    def __init__(self, name="None"):
        self.name = name
        self.gold_number = 0
        self.sys_number = 0
        self.matched_number = 0
        self.overlapped_number = 0
        self.unoverlapped_number = 0

    def print(self):
        print(f"name: {self.name}")
        print(f"gold number: {self.gold_number}")
        print(f"sys number: {self.sys_number}")
        print(f"matched number: {self.matched_number}")
        print(f"overlapped number: {self.overlapped_number}")
        print(f"unoverlapped number: {self.unoverlapped_number}")


class EvalMetric():
    def __init__(self, name="None"):
        self.name = name
        self.matched, self.sys, self.gold = 0, 0, 0
        self.p = self.r = self.f = 0.0

    def compute_prf(self):
        try:
            self.p = 100.0 * self.matched / self.sys
        except:
            self.p = 0.0
        try:
            self.r = 100.0 * self.matched / self.gold
        except:
            self.r = 0.0
        try:
            self.f = 2.0 * self.p * self.r / (self.p + self.r)
        except:
            self.f = 0.0
        print("="*5, self.name, "="*5)
        print("Precision:", self.matched, '/', self.sys, '=', self.p)
        print("Recall:", self.matched, '/', self.gold, '=', self.r)
        print("F1 score:", self.f)


def analyze_error_prediction_matrix(samples):
    agent_binary, target_binary, all_binary = EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")
    agent_proportional, target_proportional, all_proportional = \
        EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")
    agent_exact, target_exact, all_exact = EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")

    holder_counter = CountMetric("Holder")
    target_counter = CountMetric("Target")

    for sample in samples:
        gold_orl, sys_orl = sample.gold_orl, sample.sys_orl
        # dict s-e: label
        dict_gold_orl, dict_sys_orl = OrderedDict(), OrderedDict()
        for g_orl in gold_orl:  # construct the expression-argument tuples
            # print(g_orl)
            dse_s, dse_e, s, e, label = g_orl
            expression = str(dse_s) + '-' + str(dse_e)
            argument = (s, e, label)
            if expression not in dict_gold_orl:
                dict_gold_orl[expression] = []
                dict_gold_orl[expression].append(argument)
            else:
                dict_gold_orl[expression].append(argument)
        for s_orl in sys_orl:  # construct the expression-argument tuples
            dse_s, dse_e, s, e, label = s_orl
            expression = str(dse_s) + '-' + str(dse_e)
            argument = (s, e, label)
            if expression not in dict_sys_orl:
                dict_sys_orl[expression] = []
                dict_sys_orl[expression].append(argument)
            else:
                dict_sys_orl[expression].append(argument)

        for expression in dict_gold_orl:  # compute the gold
            for argument in dict_gold_orl[expression]:
                s, e, label = argument
                all_binary.gold += 1
                all_proportional.gold += 1
                all_exact.gold += 1
                if label == AGENT:
                    agent_binary.gold += 1
                    agent_proportional.gold += 1
                    agent_exact.gold += 1
                else:
                    assert label == TARGET
                    target_binary.gold += 1
                    target_proportional.gold += 1
                    target_exact.gold += 1
                if label == AGENT:
                    holder_counter.gold_number += 1
                else:
                    assert label == TARGET
                    target_counter.gold_number += 1

        for expression in dict_sys_orl:  # compute the sys
            for argument in dict_sys_orl[expression]:
                s, e, label = argument
                all_binary.sys += 1
                all_proportional.sys += 1
                all_exact.sys += 1
                if label == AGENT:
                    agent_binary.sys += 1
                    agent_proportional.sys += 1
                    agent_exact.sys += 1
                else:
                    assert label == TARGET
                    target_binary.sys += 1
                    target_proportional.sys += 1
                    target_exact.sys += 1
                if label == AGENT:
                    holder_counter.sys_number += 1
                else:
                    assert label == TARGET
                    target_counter.sys_number += 1

        for expression in dict_sys_orl:  # compute the sys
            if expression not in dict_gold_orl:  # debug: some gold orl has no argument, only expression
                # print(sample.sentence)
                for argument in dict_sys_orl[expression]:
                    s, e, label = argument
                    if label == AGENT:
                        holder_counter.unoverlapped_number += 1
                    else:
                        assert label == TARGET
                        target_counter.unoverlapped_number += 1
                continue
            gold_arguments = dict_gold_orl[expression]
            for argument in dict_sys_orl[expression]:
                s, e, label = argument
                if argument in gold_arguments:  # exact
                    all_binary.matched += 1
                    all_proportional.matched += 1
                    all_exact.matched += 1
                    if label == AGENT:
                        agent_binary.matched += 1
                        agent_proportional.matched += 1
                        agent_exact.matched += 1
                        holder_counter.matched_number += 1
                    else:
                        assert label == TARGET
                        target_binary.matched += 1
                        target_proportional.matched += 1
                        target_exact.matched += 1
                        target_counter.matched_number += 1
                else:
                    # binary
                    find = False
                    for index in range(s, e + 1):
                        for gold_arg in gold_arguments:
                            g_s, g_e, g_label = gold_arg
                            if g_label == label:
                                if g_s <= index <= g_e:
                                    all_binary.matched += 1
                                    if label == AGENT:
                                        agent_binary.matched += 1
                                    else:
                                        target_binary.matched += 1
                                    find = True
                                    break
                        if find is True:
                            if label == AGENT:
                                holder_counter.overlapped_number += 1
                            else:
                                assert label == TARGET
                                target_counter.overlapped_number += 1
                            break
                    if find is False:
                        if label == AGENT:
                            holder_counter.unoverlapped_number += 1
                        else:
                            assert label == TARGET
                            target_counter.unoverlapped_number += 1
                    # proportional
                    list_of_proportional = []
                    for gold_argument in dict_gold_orl[expression]:
                        g_s, g_e, g_label = gold_argument
                        matched_positions = 0
                        if label != g_label:
                            pass
                        else:
                            for position in range(g_s, g_e + 1):
                                if s <= position <= e:
                                    matched_positions += 1
                            list_of_proportional.append(1.0 * matched_positions / (g_e - g_s + 1))
                    if len(list_of_proportional) > 0:  # matched a gold argument
                        all_proportional.matched += max(list_of_proportional)
                        if label == AGENT:
                            agent_proportional.matched += max(list_of_proportional)
                        else:
                            target_proportional.matched += max(list_of_proportional)
        # for expression in dict_gold_orl:  # compute the proportion
        #     if expression not in dict_sys_orl:
        #         continue
        #     sys_arguments = dict_sys_orl[expression]
        #     for gold_argument in dict_gold_orl[expression]:
        #         g_s, g_e, g_label = gold_argument
        #
        #         list_of_proportional = []
        #         for sys_arg in sys_arguments:
        #             s, e, label = sys_arg
        #             matched_positions = 0
        #             if label != g_label:
        #                 pass
        #             else:
        #                 for position in range(g_s, g_e + 1):
        #                     if s <= position <= e:
        #                         matched_positions += 1
        #                 list_of_proportional.append(1.0 * matched_positions / (g_e - g_s + 1))
        #         if len(list_of_proportional) > 0:  # matched a gold argument
        #             all_proportional.matched += max(list_of_proportional)
        #             if g_label == AGENT:
        #                 agent_proportional.matched += max(list_of_proportional)
        #             else:
        #                 target_proportional.matched += max(list_of_proportional)

    print("="*15, 'Binary Metric', "="*15)
    agent_binary.compute_prf()
    target_binary.compute_prf()
    all_binary.compute_prf()

    print("="*15, 'Proportional Metric', "="*15)
    agent_proportional.compute_prf()
    target_proportional.compute_prf()
    all_proportional.compute_prf()

    print("="*15, 'Exact Metric', "="*15)
    agent_exact.compute_prf()
    target_exact.compute_prf()
    all_exact.compute_prf()

    print("="*15, "holder/target counts", "="*15)
    holder_counter.print()
    target_counter.print()

    return ((agent_binary, target_binary, all_binary),
            (agent_proportional, target_proportional, all_proportional),
            (agent_exact, target_exact, all_exact))


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    data = load_eval_data(input_file_path)
    analyze_error_prediction_matrix(data)
