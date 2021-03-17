"""
This version of baseline is from lsgn-pytorch/src/baseline-re-check-w-mask-full-embedding
Mdify:
    2020.07.22: hard MTL cons, soft MTL dep, and BERT
    2020.07.22: modify the dependency parsing module
    2020.07.13: add the implicit dependency representations (IDR).
    2020.07.09: separate the layer of the two BiLSTMs, and use bigger syntax batch
    2020.07.08: try to add the constituent label into the orl spans
    2020.07.07: try MTL and GCN, Based on GCN 3.0: add the dropout over the gcn input
    2020.07.04: add the GCN graph
    2020.06.24: add the constituent spans
    2020.06.15: add joint and mtl into the config
    2020.06.15: 3.0 fix the error of "RuntimeError: copy_if failed to synchronize: device-side assert triggered"
    2020.06.03: optimize the code
    2020.06.03: move to python 3.7
"""
# -*- coding: utf-8 -*-
import argparse
import time
import sys
import os
import numpy
import shutil
import torch
import random
import neural_srl.shared.reader as reader

from neural_srl.shared.reader import get_pretrained_embeddings
from neural_srl.shared import configuration
from neural_srl.shared.tagger_data import TaggerData, mix_training_data
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.shared.evaluation import SRLEvaluator
from neural_srl.shared.constituent_extraction import load_constituent_trees
from neural_srl.shared.syntactic_extraction import SyntacticCONLL
from neural_srl.shared.inference_utils import srl_decode
from neural_srl.shared.srl_eval_utils import compute_orl_fl, load_gold_orl_span


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = "cpu"


def adjust_learning_rate(optimizer, last_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = last_lr * 0.999
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate_tagger(model, batched_dev_data, orl_cons, auto_dep_data, dev_gold_spans, label_dict, config,
                    evaluator, writer, global_step):
    print("Evaluating")
    dev_loss = 0
    srl_predictions = []
    dse_predictions = []

    model.eval()
    with torch.no_grad():
        matched_pred = sys_pred = gold_pred = 0
        matched_argu = sys_argu = gold_argu = 0
        matched_srl = sys_srl = gold_srl = 0

        matched_dse_starts = sys_dse_starts = gold_dse_starts = 0
        matched_dse_ends = sys_dse_ends = gold_dse_ends = 0
        matched_arg_starts = sys_arg_starts = gold_arg_starts = 0
        matched_arg_ends = sys_arg_ends = gold_arg_ends = 0
        for i, batched_tensor in enumerate(batched_dev_data):
            sent_lengths, word_indexes, char_indexes, \
            dse_starts, dse_ends, dse_num, dse_starts_one_hot, dse_ends_one_hot, \
            arg_starts, arg_ends, arg_num, arg_starts_one_hot, arg_ends_one_hot, \
            orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl, orl_num, \
            sent_words = batched_tensor
            cons_trees = [orl_cons[' '.join(words)] for words in sent_words]
            auto_dep_trees = [auto_dep_data[' '.join(words)] for words in sent_words]

            if args.gpu:
                word_indexes, char_indexes, \
                dse_starts, dse_ends, dse_starts_one_hot, dse_ends_one_hot, \
                arg_starts, arg_ends, arg_starts_one_hot, arg_ends_one_hot, \
                orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl = \
                    word_indexes.cuda(), char_indexes.cuda(), \
                    dse_starts.cuda(), dse_ends.cuda(), dse_starts_one_hot.cuda(), dse_ends_one_hot.cuda(), \
                    arg_starts.cuda(), arg_ends.cuda(), arg_starts_one_hot.cuda(), arg_ends_one_hot.cuda(), \
                    orl_dse_starts.cuda(), orl_dse_ends.cuda(), orl_arg_starts.cuda(), orl_arg_ends.cuda(), orl_arg_srl.cuda()

            hot_start = False
            # hot_start = False if epoch == 0 else False
            predicated_dict, loss = model.forward(
                sent_lengths,
                (sent_words, word_indexes, char_indexes),
                (dse_starts, dse_ends, dse_num, dse_starts_one_hot, dse_ends_one_hot),
                (arg_starts, arg_ends, arg_num, arg_starts_one_hot, arg_ends_one_hot),
                (orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl, orl_num),
                cons_trees,
                auto_dep_trees,
                hot_start=hot_start)

            dev_loss += float(loss)

            matched_pred += predicated_dict["matched_dse_num"]
            sys_pred += predicated_dict["sys_dse_num"]
            gold_pred += predicated_dict["gold_dse_num"]

            matched_argu += predicated_dict["matched_argu_num"]
            sys_argu += predicated_dict["sys_argu_num"]
            gold_argu += predicated_dict["gold_argu_num"]

            matched_srl += predicated_dict["matched_srl_num"]
            sys_srl += predicated_dict["sys_srl_num"]
            gold_srl += predicated_dict["gold_srl_num"]

            matched_dse_starts += predicated_dict["matched_dse_starts"]
            sys_dse_starts += predicated_dict["pred_dse_starts"]
            gold_dse_starts += predicated_dict["gold_dse_starts"]

            matched_dse_ends += predicated_dict["matched_dse_ends"]
            sys_dse_ends += predicated_dict["pred_dse_ends"]
            gold_dse_ends += predicated_dict["gold_dse_ends"]

            matched_arg_starts += predicated_dict["matched_arg_starts"]
            sys_arg_starts += predicated_dict["pred_arg_starts"]
            gold_arg_starts += predicated_dict["gold_arg_starts"]

            matched_arg_ends += predicated_dict["matched_arg_ends"]
            sys_arg_ends += predicated_dict["pred_arg_ends"]
            gold_arg_ends += predicated_dict["gold_arg_ends"]

            decoded_predictions = srl_decode(sent_lengths, predicated_dict,
                                             label_dict.idx2str, config)
            #
            if "srl" in decoded_predictions:
                srl_predictions.extend(decoded_predictions["srl"])
            dse_predictions.extend(decoded_predictions["dse"])
    try:
        print("="*10, "Evaluating", "="*10)
        print("-"*10, "DSE ARG ORL Info", "-"*10)
        print("DSE num of (matched, sys, and gold)", matched_pred, sys_pred, gold_pred)
        print("ARG num of (matched, sys, and gold)", matched_argu, sys_argu, gold_argu)
        print("ORL num pf (matched, sys, and gold)", matched_srl, sys_srl, gold_srl)
        print("-"*10, "Boundary Info", "-"*10)
        print("DSE STARTS num of (matched, sys, gold)", matched_dse_starts, sys_dse_starts, gold_dse_starts)
        print("DSE END num of (matched, sys, gold)", matched_dse_ends, sys_dse_ends, gold_dse_ends)
        print("ARG STARTS num of (matched, sys, gold)", matched_arg_starts, sys_arg_starts, gold_arg_starts)
        print("ARG END num of (matched, sys, gold)", matched_arg_ends, sys_arg_ends, gold_arg_ends)

        print("-"*10, "DSE ARG ORL (P, R, F1)", "-"*10)
        pred_p = 100.0 * matched_pred / sys_pred
        pred_r = 100.0 * matched_pred / gold_pred
        pred_f = 2.0 * pred_p * pred_r / (pred_p + pred_r)
        print("DSE P R F1", pred_p, pred_r, pred_f)
        #
        argu_p = 100.0 * matched_argu / sys_argu
        argu_r = 100.0 * matched_argu / gold_argu
        argu_f = 2.0 * argu_p * argu_r / (argu_p + argu_r)
        print("ARG P R F1", argu_p, argu_r, argu_f)
        #
        srl_p = 100.0 * matched_srl / sys_srl
        srl_r = 100.0 * matched_srl / gold_srl
        srl_f = 2.0 * srl_p * srl_r / (srl_p + srl_r)
        print("ORL P R F1", srl_p, srl_r, srl_f)

        print("-"*10, "Boundary (P, R, F1)", "-"*10)
        dse_starts_p = 100.0 * matched_dse_starts / sys_dse_starts
        dse_starts_r = 100.0 * matched_dse_starts / gold_dse_starts
        dse_starts_f = 2.0 * dse_starts_p * dse_starts_r / (dse_starts_p + dse_starts_r)
        print("DSE starts predicted info (p, r, f)", dse_starts_p, dse_starts_r, dse_starts_f)
        dse_ends_p = 100.0 * matched_dse_ends / sys_dse_ends
        dse_ends_r = 100.0 * matched_dse_ends / gold_dse_ends
        dse_ends_f = 2.0 * dse_ends_p * dse_ends_r / (dse_ends_p + dse_ends_r)
        print("DSE ends predicted info (p, r, f)", dse_ends_p, dse_ends_r, dse_ends_f)

        arg_starts_p = 100.0 * matched_arg_starts / sys_arg_starts
        arg_starts_r = 100.0 * matched_arg_starts / gold_arg_starts
        arg_starts_f = 2.0 * arg_starts_p * arg_starts_r / (arg_starts_p + arg_starts_r)
        print("ARG starts predicted info (p, r, f)", arg_starts_p, arg_starts_r, arg_starts_f)
        arg_ends_p = 100.0 * matched_arg_ends / sys_arg_ends
        arg_ends_r = 100.0 * matched_arg_ends / gold_arg_ends
        arg_ends_f = 2.0 * arg_ends_p * arg_ends_r / (arg_ends_p + arg_ends_r)
        print("ARG ends predicted info (p, r, f)", arg_ends_p, arg_ends_r, arg_ends_f)
    except Exception:
        print("pass because of zero division")

    print('Dev loss={:.6f}'.format(dev_loss / len(batched_dev_data)))

    conll_f1 = compute_orl_fl(dev_gold_spans, dse_predictions, srl_predictions)
    # sentences, gold_srl, gold_ner = zip(*eval_data)
    # # Summarize results, evaluate entire dev set.
    # precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp =\
    #     (compute_srl_f1(sentences, gold_srl, srl_predictions, args.gold, config.span_based))

    if conll_f1 > evaluator.best_accuracy:
        evaluator.best_accuracy = conll_f1
        evaluator.has_best = True
    else:
        evaluator.has_best = False
    writer.write('{}\t{}\t{:.6f}\t{:.2f}\t{:.2f}\n'.format(
        global_step, time.strftime("%Y-%m-%d %H:%M:%S"), float(dev_loss),
        float(conll_f1), float(evaluator.best_accuracy)))
    writer.flush()
    sys.stdout.flush()
    if evaluator.has_best:
        model.save(os.path.join(args.model, 'model'))


def train_tagger(args):
    # get the parse configuration
    config = configuration.get_config(args.config)
    config.span_based = args.span == "span"
    # set random seeds of numpy and torch
    random.seed(666)
    numpy.random.seed(666)
    torch.manual_seed(666)
    # GPU of pytorch
    gpu = torch.cuda.is_available()
    if args.gpu and gpu:
        print("GPU available? {}\t and GPU ID is : {}".format(gpu, args.gpu))
        # set pytorch.cuda's random seed
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.cuda.manual_seed(666)
    # set pytorch print precision
    torch.set_printoptions(precision=20)
    # set the default number of threads
    torch.set_num_threads(4)

    with Timer('Data loading'):
        # Collect Data for SRL.
        data = TaggerData(
            config,
            *reader.get_srl_data(config, args.train, args.dev,
                                 args.cons_trees, args.auto_cons_trees,
                                 args.dep_trees, args.auto_dep_trees
                                 )
        )
        # Generate SRL evaluator for Dev data
        """Actually, this evaluator has been abandoned, and the only function is to store the highest accuracy."""
        evaluator = SRLEvaluator()
        batched_dev_data = data.get_development_data(
            batch_size=config.dev_batch_size)
        dev_gold_spans = load_gold_orl_span(args.dev)
        print('Dev data has {} batches.'.format(len(batched_dev_data)))

    with Timer('Preparation'):
        if not os.path.isdir(args.model):
            print('Directory {} does not exist. Creating new.'.format(
                args.model))
            os.makedirs(args.model)
        else:
            if len(os.listdir(args.model)) > 0:
                print(
                    '[WARNING] Log directory {} is not empty, previous checkpoints might be overwritten'
                    .format(args.model))
        shutil.copyfile(args.config, os.path.join(args.model, 'config'))
        # Save word and label dict to model directory.
        data.word_dict.save(os.path.join(args.model, 'word_dict'))
        data.char_dict.save(os.path.join(args.model, 'char_dict'))
        data.pos_dict.save(os.path.join(args.model, 'pos_dict'))
        data.label_dict.save(os.path.join(args.model, 'label_dict'))
        data.cons_label_dict.save(os.path.join(args.model, 'cons_label_dict'))
        data.dep_label_dict.save(os.path.join(args.model, 'dep_label_dict'))

        writer = open(os.path.join(args.model, 'checkpoints.tsv'), 'w')
        writer.write(
            'step\tdatetime\tdev_loss\tdev_accuracy\tbest_dev_accuracy\n')

    with Timer('Building NN model'):
        model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
        if args.gpu:
            print("BiLSTMTaggerModel initialize with GPU!")
            model = model.to(device)
            if args.gpu != "" and not torch.cuda.is_available():
                raise Exception("No GPU Found!")
        print(model)  # print our model

    i, global_step, epoch, train_loss, epoch_train_loss = 0, 0, 0, 0.0, 0.0
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    last_lr = float(config.learning_rate)
    no_more_better_performance = 0
    lr_decay_steps = int(config.decay_steps)
    # initialize the model parameter optimizer
    optimizer = torch.optim.Adam(parameters,
                                 lr=last_lr,
                                 weight_decay=float(config.weight_decay))
    while global_step <= 320000:  # epoch < config.max_epochs
        initial_time = time.time()
        with Timer("Epoch%d" % epoch) as timer:
            model.train()
            mixed_data = []
            train_data = data.get_training_data(include_last_batch=True)
            mixed_data.append(train_data)
            if config.mtl_cons:
                cons_data = data.get_cons_data(include_last_batch=True)
                mixed_data.append(cons_data)
            if config.mtl_dep:
                dep_train_data = data.get_dep_training_data(include_last_batch=True)
                mixed_data.append(dep_train_data)
            batched_data = mix_training_data(mixed_data)
            for batch_data in batched_data:  # for each batch in the training corpus
                if config.mtl_cons and not config.mtl_dep:
                    batched_tensor, sub_batched_cons = batch_data
                elif not config.mtl_cons and config.mtl_dep:
                    batched_tensor, batched_dep_tensor = batch_data
                elif config.mtl_cons and config.mtl_dep:
                    batched_tensor, sub_batched_cons, batched_dep_tensor = batch_data
                else:
                    assert not config.mtl_cons and not config.mtl_dep
                    batched_tensor = batch_data[0]
                if config.mtl_cons:  # do the MTL cons
                    optimizer.zero_grad()
                    # for batched_cons in sub_batched_cons:
                    cons_sent_lengths, cons_word_indexes, cons_char_indexes,\
                        cons_starts, cons_ends, cons_labels, cons_num,\
                        cons_starts_one_hot, cons_ends_one_hot = sub_batched_cons

                    if args.gpu:
                        cons_word_indexes, cons_char_indexes, cons_starts, cons_ends, cons_labels,\
                        cons_starts_one_hot, cons_ends_one_hot = \
                        cons_word_indexes.cuda(), cons_char_indexes.cuda(), \
                        cons_starts.cuda(), cons_ends.cuda(), cons_labels.cuda(), \
                        cons_starts_one_hot.cuda(), cons_ends_one_hot.cuda()

                    cons_loss = model.cons_forward(
                        ((cons_sent_lengths, cons_word_indexes, cons_char_indexes),
                         (cons_starts, cons_ends, cons_labels, cons_num),
                         (cons_starts_one_hot, cons_ends_one_hot)))
                    cons_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                if config.mtl_dep:
                    # dep forward
                    word_indexes, char_indexes, mask, lengths, heads, labels = batched_dep_tensor
                    if args.gpu:
                        word_indexes, char_indexes = word_indexes.cuda(), char_indexes.cuda()

                    optimizer.zero_grad()
                    _, dep_loss = model.dep_forward(
                        lengths, (word_indexes, char_indexes),
                        None, None, None, None,
                        (heads, labels)
                    )

                    dep_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                sent_lengths, word_indexes, char_indexes, \
                dse_starts, dse_ends, dse_num, dse_starts_one_hot, dse_ends_one_hot, \
                arg_starts, arg_ends, arg_num, arg_starts_one_hot, arg_ends_one_hot, \
                orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl, orl_num, \
                sent_words = batched_tensor
                auto_cons_trees = [data.auto_cons_data[' '.join(words)] for words in sent_words]
                auto_dep_trees = [data.auto_dep_data[' '.join(words)] for words in sent_words]
                if args.gpu:
                    word_indexes, char_indexes, \
                    dse_starts, dse_ends, dse_starts_one_hot, dse_ends_one_hot, \
                    arg_starts, arg_ends, arg_starts_one_hot, arg_ends_one_hot, \
                    orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl = \
                        word_indexes.cuda(), char_indexes.cuda(), \
                        dse_starts.cuda(), dse_ends.cuda(), dse_starts_one_hot.cuda(), dse_ends_one_hot.cuda(), \
                        arg_starts.cuda(), arg_ends.cuda(), arg_starts_one_hot.cuda(), arg_ends_one_hot.cuda(), \
                        orl_dse_starts.cuda(), orl_dse_ends.cuda(), orl_arg_starts.cuda(), orl_arg_ends.cuda(), orl_arg_srl.cuda()

                optimizer.zero_grad()
                hot_start = False
                # hot_start = False if epoch == 0 else False
                print("gold dse_starts\n", dse_starts)
                print("gold dse_ends\n", dse_ends)
                print("gold arg_starts\n", arg_starts)
                print("gold arg_ends\n", arg_ends)
                predicated_dict, loss = model.forward(
                    sent_lengths,
                    (sent_words, word_indexes, char_indexes),
                    (dse_starts, dse_ends, dse_num, dse_starts_one_hot, dse_ends_one_hot),
                    (arg_starts, arg_ends, arg_num, arg_starts_one_hot, arg_ends_one_hot),
                    (orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl, orl_num),
                    auto_cons_trees,
                    auto_dep_trees,
                    hot_start=hot_start)
                loss.backward()
                exit()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += float(
                    loss.detach()
                )  # should be tensor not Variable, avoiding the graph accumulates
                epoch_train_loss += float(loss.detach())

                i += 1
                global_step += 1
                if global_step % lr_decay_steps == 0:
                    last_lr = adjust_learning_rate(optimizer, last_lr)
                if i % 250 == 0:
                    total_time = time.time() - initial_time
                    timer.tick(
                        "{} training steps, loss={:.3f}, steps/s={:.2f}, learning rate={}".
                        format(global_step, float(train_loss / i),
                               float(global_step / total_time), last_lr))
                    train_loss = 0.0
                    i = 0
                if global_step % 10 == 0:  # evaluate for every 100 steps
                    with Timer('Evaluation'):
                        evaluate_tagger(model, batched_dev_data,
                                        data.auto_cons_data, data.auto_dep_data,
                                        dev_gold_spans,
                                        data.label_dict, config, evaluator, writer,
                                        global_step)
                    model.train()
                    if evaluator.has_best is True:
                        no_more_better_performance = 0
                    else:
                        no_more_better_performance += 1
                        if no_more_better_performance >= 100:
                            print(
                                "no more better performance since the past 100 epochs!"
                            )
                            exit()

            train_loss = train_loss / i
            print("Epoch {}, steps={}, loss={:.3f}, epoch loss={:.3f}".format(
                epoch, i, float(train_loss), epoch_train_loss))

            i = 0
            epoch += 1
            train_loss = 0.0
            epoch_train_loss = 0.0

    writer.close()
    # Done. :)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=str,
        default='',
        required=True,
        help='Config file for the neural architecture and hyper-parameters.')

    parser.add_argument(
        '--span',
        type=str,
        default='span',
        required=True,
        help='Whether current experiments is for span-based SRL. Default True (span-based SRL).'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='',
        required=True,
        help='Path to the directory for saving model and checkpoints.')

    parser.add_argument(
        '--train',
        type=str,
        default='',
        required=True,
        help='Path to the training data, which is a single file in sequential tagging format.'
    )

    parser.add_argument(
        '--dev',
        type=str,
        default='',
        required=True,
        help='Path to the devevelopment data, which is a single file in the sequential tagging format.'
    )

    parser.add_argument(
        '--cons_trees',
        type=str,
        default='',
        required=False,
        help='Path to the constituent spans data, which is a single file in the sequential tagging format.'
    )

    parser.add_argument(
        '--auto_cons_trees',
        type=str,
        default='',
        required=False,
        help='Path to the constituent spans data, which is a single file in the sequential tagging format.'
    )

    parser.add_argument(
        '--dep_trees',
        type=str,
        default="",
        required=False,
        help="Path to the dependency trees data, which is a single file in CoNLL format."
    )

    parser.add_argument(
        '--auto_dep_trees',
        type=str,
        default='',
        required=False,
        help='Path to the automatic orl dependency trees.'
    )

    parser.add_argument(
        '--gold',
        type=str,
        default='',
        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default="",
        help='(Optional) A argument that specifies the GPU id. Default use the cpu')

    parser.add_argument(
        '--info',
        type=str,
        default="",
        help='(Optional) A additional information that specify this program.')
    args = parser.parse_args()
    train_tagger(args)
