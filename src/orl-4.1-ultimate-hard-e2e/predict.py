''' Predict and output scores.

   - Reads model param file.
   - Runs data.
   - Remaps label indices.
   - Outputs protobuf file.
'''

import argparse
import torch
import os
import neural_srl.shared.reader as reader

from neural_srl.shared import configuration
from neural_srl.shared.constants import UNKNOWN_TOKEN, PADDING_TOKEN
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.shared.inference_utils import srl_decode
from neural_srl.shared.srl_eval_utils import compute_orl_fl, load_gold_orl_span, output_predicted_file
from neural_srl.shared.reader import load_eval_data
from neural_srl.shared.constituent_extraction import load_constituent_trees
from neural_srl.shared.reader import get_pretrained_embeddings

# from neural_srl.shared.numpy_saver import NumpySaver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
        help='Path to the model directory.'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='',
        required=True,
        help='Path to the input file path (sequetial tagging format).')

    parser.add_argument(
        '--orl_cons',
        type=str,
        default='',
        required=True,
        help='Path to the constituent spans data, which is a single file in the sequential tagging format.'
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
        required=True,
        help='Path to the input file path (sequetial tagging format).')

    parser.add_argument(
        '--task',
        type=str,
        help='Training task (srl or propid). Default is srl.',
        default='srl',
        choices=['srl', 'propid']
    )

    parser.add_argument(
        '--mate_tools_senses',
        type=str,
        default='',
        required=False,
        help='(Optional) Path to the mate-tools senses file.'
    )

    parser.add_argument(
        '--inputprops',
        type=str,
        default='',
        help='(Optional) Path to the predicted predicates in CoNLL format. Ignore if using gold predicates.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='(Optional) Path for output predictions.'
    )

    parser.add_argument(
        '--outputprops',
        type=str,
        default='',
        help='(Optional) Path for output predictions in CoNLL format. Only used when task is {propid}.'
    )

    parser.add_argument(
        '--proto',
        type=str,
        default='',
        help='(Optional) Path to the proto file path (for reusing predicted scores).'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default="",
        help='(Optional) A argument that specifies the GPU id. Default use the cpu'
    )

    args = parser.parse_args()
    config = configuration.get_config(os.path.join(args.model, 'config'))

    config.span_based = args.span == "span"

    # Detect available ensemble models.
    task = "srl"
    num_ensemble_models = 1
    if num_ensemble_models == 1:
        print('Using single model.')
    else:
        print('Using an ensemble of {} models'.format(num_ensemble_models))

    ensemble_scores = None
    for i in range(num_ensemble_models):
        if num_ensemble_models == 1:
            model_path = os.path.join(args.model, 'model')
            word_dict_path = os.path.join(args.model, 'word_dict')
        else:
            model_path = os.path.join(args.model, 'model{}.npz'.format(i))
            word_dict_path = os.path.join(args.model, 'word_dict{}'.format(i))
        pos_dict_path = os.path.join(args.model, 'pos_dict')
        char_dict_path = os.path.join(args.model, 'char_dict')
        label_dict_path = os.path.join(args.model, 'label_dict')
        dep_dict_path = os.path.join(args.model, 'dep_label_dict')
        cons_label_dict_path = os.path.join(args.model, 'cons_label_dict')
        non_terminal_dict_path = os.path.join(args.model, 'cons_non_terminal_dict')
        print("model dict path: {}, word dict path: {}, char dict path:{}, label dict path: {}".format(
            model_path, word_dict_path, char_dict_path, label_dict_path))

        with Timer('Data loading'):
            print('Task: {}'.format(task))
            allow_new_words = True
            print('Allow new words in test data: {}'.format(allow_new_words))
            # Load word and tag dictionary
            word_dict = Dictionary(
                padding_token=PADDING_TOKEN,
                unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            char_dict = Dictionary(
                padding_token=PADDING_TOKEN,
                unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            pos_dict = Dictionary()
            label_dict = Dictionary()
            dep_label_dict = Dictionary()
            cons_label_dict = Dictionary()
            non_terminal_dict = Dictionary()
            word_dict.load(word_dict_path)
            char_dict.load(char_dict_path)
            pos_dict.load(pos_dict_path)
            label_dict.load(label_dict_path)
            dep_label_dict.load(dep_dict_path)
            cons_label_dict.load(cons_label_dict_path)
            non_terminal_dict.load(non_terminal_dict_path)
            char_dict.accept_new, label_dict.accept_new, cons_label_dict.accept_new = False, False, False

            # with Timer("Loading orl data constituent trees"):
            #     word_embeddings = get_pretrained_embeddings(config.word_embedding)  # get pre-trained embeddings
            #     orl_cons, pos_dict, word_dict = load_constituent_trees(args.orl_cons, word_embeddings,
            #                                                            word_dict, pos_dict)

            data = TaggerData(config,
                              [], [], [],
                              [], [],
                              [], [],
                              word_dict, char_dict, pos_dict,
                              label_dict,
                              cons_label_dict, dep_label_dict,
                              non_terminal_dict,
                              None, None)
            # Load test data.
            if task == 'srl':
                test_sentences, auto_cons, auto_dep, emb, emb_shapes = reader.get_srl_test_data(
                    args.input, config,
                    data.word_dict, data.char_dict, data.label_dict,
                    args.orl_cons, args.auto_dep_trees,
                    data.dep_label_dict,
                    allow_new_words)
                eval_data = load_eval_data(args.input)

            print('Read {} sentences.'.format(len(test_sentences)))
            # Add pre-trained embeddings for new words in the test data.
            # if allow_new_words:
            data.word_embeddings = emb
            data.word_embedding_shapes = emb_shapes
            # Batching.
            test_data = data.get_test_data(test_sentences,
                                           batch_size=config.dev_batch_size)
            test_gold_spans = load_gold_orl_span(args.input)

        with Timer('Model building and loading'):
            model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
            model.load(model_path)
            print(model)
            if args.gpu:
                print("Initialize the model with GPU!")
                model = model.cuda()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

        print("model parameters", count_parameters(model))

        with Timer('Running model'):
            dev_loss = 0.0
            srl_predictions, dse_predictions, cons_predictions, argu_cons_predictions = [], [], [], []
            sys_args = []

            model.eval()
            with torch.no_grad():  # Eval don't need the grad
                matched_pred = sys_pred = gold_pred = 0
                matched_argu = sys_argu = gold_argu = 0
                matched_srl = sys_srl = gold_srl = 0

                matched_dse_starts = sys_dse_starts = gold_dse_starts = 0
                matched_dse_ends = sys_dse_ends = gold_dse_ends = 0
                matched_arg_starts = sys_arg_starts = gold_arg_starts = 0
                matched_arg_ends = sys_arg_ends = gold_arg_ends = 0

                for i, batched_tensor in enumerate(test_data):
                    sent_lengths, word_indexes, char_indexes, \
                    dse_starts, dse_ends, dse_num, dse_starts_one_hot, dse_ends_one_hot, \
                    arg_starts, arg_ends, arg_num, arg_starts_one_hot, arg_ends_one_hot, \
                    orl_dse_starts, orl_dse_ends, orl_arg_starts, orl_arg_ends, orl_arg_srl, orl_num, \
                    sent_words = batched_tensor
                    # cons_trees = [auto_cons[' '.join(words)] for words in sent_words]
                    cons_trees = None if config.use_cons_gcn is False else [auto_cons[' '.join(words)] for words in sent_words]
                    # auto_dep_trees = [auto_dep[' '.join(words)] for words in sent_words]
                    auto_dep_trees = None

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

                    decoded_predictions = srl_decode(sent_lengths,
                                                     predicated_dict,
                                                     label_dict.idx2str,
                                                     config,
                                                     cons_idx2label=cons_label_dict.idx2str)
                    #
                    if "srl" in decoded_predictions:
                        srl_predictions.extend(decoded_predictions["srl"])
                        dse_predictions.extend(decoded_predictions["dse"])
                        sys_args.extend(decoded_predictions["arg"])
                    if "cons" in decoded_predictions:
                        cons_predictions.extend(decoded_predictions["cons"])
                    if "argu_cons" in decoded_predictions:
                        argu_cons_predictions.extend(decoded_predictions["argu_cons"])
            try:
                print("=" * 10, "Evaluating", "=" * 10)
                print("-" * 10, "DSE ARG ORL Info", "-" * 10)
                print("DSE num of (matched, sys, and gold)", matched_pred, sys_pred, gold_pred)
                print("ARG num of (matched, sys, and gold)", matched_argu, sys_argu, gold_argu)
                print("ORL num pf (matched, sys, and gold)", matched_srl, sys_srl, gold_srl)
                print("-" * 10, "Boundary Info", "-" * 10)
                print("DSE STARTS num of (matched, sys, gold)", matched_dse_starts, sys_dse_starts, gold_dse_starts)
                print("DSE END num of (matched, sys, gold)", matched_dse_ends, sys_dse_ends, gold_dse_ends)
                print("ARG STARTS num of (matched, sys, gold)", matched_arg_starts, sys_arg_starts, gold_arg_starts)
                print("ARG END num of (matched, sys, gold)", matched_arg_ends, sys_arg_ends, gold_arg_ends)

                print("-" * 10, "DSE ARG ORL (P, R, F1)", "-" * 10)
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

                print("-" * 10, "Boundary (P, R, F1)", "-" * 10)
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
                # return

            compute_orl_fl(test_gold_spans, dse_predictions, srl_predictions, is_gold_dse=config.use_gold_predicates)
            output_predicted_file(args.gold, test_gold_spans, srl_predictions, sys_args, outputpath=args.output,
                                  sys_cons=[cons_predictions, argu_cons_predictions])
