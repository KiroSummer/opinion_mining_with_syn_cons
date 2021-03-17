import sys
import json
from collections import OrderedDict


DSE="DSE"
TARGET="TARGET"
AGENT="AGENT"

max_dse_length, max_target_length, max_agent_length = 0, 0, 0


class orl_data():
    def __init__(self, tuples):
        self.idx = []
        self.words = []
        self.labels = []
        self.des = []
        self.des_head = []
        self.target = []
        self.agent = []
        self.orl = []
        self.init_by_typles(tuples)

    def output_to_srl_json(self):
        srl_span = []
        for span in self.orl:
            s, e, a_s, a_e, label = span
            if label == "DSE":
                continue
            count = 0
            for des_head in self.des_head:
                if s <= des_head <= e:
                    srl_span.append([des_head, a_s, a_e, label])
                    count += 1
            if count != 1:
                print(self.words, self.des_head)
            # assert count == 1
        for des_head in self.des_head:
            srl_span.append([des_head, des_head, des_head, "V"])
        output = {
            "speakers": [["-"] * len(self.words)],
            "doc_key": "S0",
            "sentences": [self.words],
            "srl": [srl_span],
            "constituents": [[]],
            "clusters": [],
            "ner": [[]]

        }
        return output

    def output_to_json(self):
        output = {
            "sentences": self.words,
            "orl": self.orl
        }
        return output

    def a_complete_span(self, des, span, label):
        # print(des, span, label)
        t = (des + span)
        t.append(label)
        # print(t)
        assert len(t) == 5
        if t[-1] == DSE:
            global max_dse_length
            dse_length = t[1] - t[0]
            max_dse_length = max_dse_length if max_dse_length > dse_length else dse_length
            self.des.append(t)
        elif t[-1] == TARGET:
            global max_target_length
            target_length = t[3] - t[2]
            max_target_length = max_target_length if max_target_length > target_length else target_length
            self.target.append(t)
        else:
            assert t[-1] == AGENT
            global max_agent_length
            agent_length = t[3] - t[2]
            max_agent_length = max_agent_length if max_agent_length > agent_length else agent_length
            self.agent.append(t)
        self.orl.append(t)

    @staticmethod
    def compose_a_span(des, spans, labels, span, l):
        assert len(span) == 2 and l != ""
        if l == DSE:
            assert len(des) == 0
            des = span
            # print("DES!!!!")
            spans.append(des)
            labels.append(l)
        else:
            spans.append(span)
            labels.append(l)
        return des

    def init_by_typles(self, tuples):
        # print(tuples)
        self.idx, self.words, self.labels = tuples[0], tuples[1], tuples[2:]
        for expression_aware_label in self.labels:
            des, spans, labels = [], [], []
            span, l = [], ''
            # print(self.label)
            for i, label in enumerate(expression_aware_label):
                if label.endswith("-*"):  # we do n't need the * that marks the ``head word''
                    label = label[:-2]
                    self.des_head.append(i)
                if label == "S-DSE":
                    self.des_head.append(i)

                if label.startswith("B"):
                    assert len(span) == 0
                    span.append(i)
                    l = label[2:]
                elif label.startswith("M"):
                    assert l == label[2:]
                elif label.startswith("E"):
                    assert l == label[2:]
                    span.append(i)
                    des = orl_data.compose_a_span(des, spans, labels, span, l)
                    span, l = [], ''
                elif label.startswith("S"):
                    span = [i, i]
                    l = label[2:]
                    # print("label", l)
                    des = orl_data.compose_a_span(des, spans, labels, span, l)
                    # print("XXX", des)
                    span, l = [], ''
                else:
                    assert label == 'O'

            assert len(spans) == len(labels)
            for s, l in list(zip(spans, labels)):
                self.a_complete_span(des, s, l)

    def write_to_json(self):
        pass


if __name__ == "__main__":
    input_filepath = sys.argv[1]

    input_data = OrderedDict()
    original_sentence_number, unique_sentence_number = 0, 0
    duplicate_sentence_number = 0
    duplicate_sentence_label_number = 0
    with open(input_filepath, 'r') as input_orl_file:
        sentence = []
        for line in input_orl_file.readlines():
            if line.strip() == "":
                original_sentence_number += 1
                tuples = list(zip(*sentence))
                sen = ' '.join(tuples[1])
                # print(sen)
                if len(input_data) != 0:
                    if sen in input_data.keys():  # if it is the same sentence
                        # print("xx")
                        if tuples[-1] not in input_data[sen][2:]:
                            input_data[sen].append(tuples[-1])
                            duplicate_sentence_number += 1
                        else:
                            print(tuples[-1], "already in previous sample", input_data[sen])
                            duplicate_sentence_label_number += 1
                    else:
                        input_data[sen] = tuples
                        unique_sentence_number += 1
                else:
                    input_data[sen] = tuples
                    unique_sentence_number += 1
                sentence = []
                continue
            tokens = line.strip().split()
            # print(tokens)
            sentence.append(tokens)
    # check for sentences appear more than once
    assert original_sentence_number == unique_sentence_number + duplicate_sentence_number +\
           duplicate_sentence_label_number
    print("original sentence number:", original_sentence_number)
    print("unique_sentence_number:", unique_sentence_number)
    print("duplicate_sentence_number:", duplicate_sentence_number)
    print("duplicate_sentence_label_number", duplicate_sentence_label_number)
    # generate_chars
    # with open(input_filepath + ".char.txt", 'w') as char_file:
    #     char = set()
    #     for sen in input_data.keys():
    #         words = sen.strip().split()
    #         for word in words:
    #             for c in word:
    #                 char.add(c)
    #     for c in char:
    #         char_file.write(c + '\n')

    sentences = set()
    for data in input_data.keys():
        sen = ' '.join(data)
        if sen not in sentences:
            sentences.add(sen)
        else:
            print(sen, "already appears!")
            # pass
    # generate orl data
    orl_dataset = []
    for data in input_data.keys():
        orl_dataset.append(orl_data(input_data[data]))
    # global max_dse_length
    # global max_target_length
    # global max_agent_length
    print("max_dse_length", max_dse_length, "max_target_length", max_target_length,
          "max_agent_length", max_agent_length)
    # output to json
    json_filename = input_filepath + '.json'
    with open(json_filename, 'w') as output_json:
        for orl in orl_dataset:
            # print(orl.output_to_json())
            output_json.write(json.dumps(orl.output_to_json()) + '\n')


