''' Predict and output scores.

   - Reads model param file.
   - Runs data.
   - Remaps label indices.
   - Outputs protobuf file.
'''

import argparse
import torch
import os

from neural_srl.shared import configuration
from neural_srl.shared import reader
from neural_srl.shared.constants import UNKNOWN_TOKEN, PADDING_TOKEN
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.shared.inference_utils import srl_decode
from neural_srl.shared.srl_eval_utils import compute_srl_f1
from neural_srl.shared.reader import load_eval_data
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

    parser.add_argument('--model',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the model directory.')

    parser.add_argument(
        '--input',
        type=str,
        default='',
        required=True,
        help='Path to the input file path (sequetial tagging format).')

    parser.add_argument('--task',
                        type=str,
                        help='Training task (srl or propid). Default is srl.',
                        default='srl',
                        choices=['srl', 'propid'])

    parser.add_argument(
        '--gold',
        type=str,
        default='',
        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).'
    )

    parser.add_argument('--mate_tools_senses',
                        type=str,
                        default='',
                        required=False,
                        help='(Optional) Path to the mate-tools senses file.')

    parser.add_argument(
        '--inputprops',
        type=str,
        default='',
        help='(Optional) Path to the predicted predicates in CoNLL format. Ignore if using gold predicates.'
    )

    parser.add_argument('--output',
                        type=str,
                        default='',
                        help='(Optional) Path for output predictions.')

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
        head_dict_path = os.path.join(args.model, 'head_dict')
        char_dict_path = os.path.join(args.model, 'char_dict')
        label_dict_path = os.path.join(args.model, 'label_dict')
        print("model dict path: {}, word dict path: {}, head dict path: {}, char dict path:{}, label dict path: {}".format(
            model_path, word_dict_path, head_dict_path, char_dict_path, label_dict_path))

        with Timer('Data loading'):
            print('Task: {}'.format(task))
            allow_new_words = True
            print('Allow new words in test data: {}'.format(allow_new_words))
            # Load word and tag dictionary
            word_dict = Dictionary(
                padding_token=PADDING_TOKEN,
                unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            head_dict = Dictionary(
                padding_token=PADDING_TOKEN,
                unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            char_dict = Dictionary(
                padding_token=PADDING_TOKEN,
                unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            label_dict = Dictionary()
            word_dict.load(word_dict_path)
            head_dict.load(head_dict_path)
            char_dict.load(char_dict_path)
            label_dict.load(label_dict_path)
            char_dict.accept_new, label_dict.accept_new = False, False

            data = TaggerData(config, [], [], [], word_dict, head_dict,
                              char_dict, label_dict, None, None)
            # Load test data.
            if task == 'srl':
                test_sentences, emb, emb_shapes = reader.get_srl_test_data(
                    args.input, config, data.word_dict, data.head_dict,
                    data.char_dict, data.label_dict, allow_new_words)
                eval_data = load_eval_data(args.input)

            print('Read {} sentences.'.format(len(test_sentences)))
            # Add pre-trained embeddings for new words in the test data.
            # if allow_new_words:
            data.word_embeddings = emb[0]
            data.head_embeddings = emb[1]
            data.word_embedding_shapes = emb_shapes[0]
            data.head_embedding_shapes = emb_shapes[1]
            # Batching.
            test_data = data.get_test_data(test_sentences,
                                           batch_size=config.dev_batch_size)

        with Timer('Model building and loading'):
            model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
            model.load(model_path)
            for param in model.parameters():
                print(param.size())
            if args.gpu:
                print("Initialize the model with GPU!")
                model = model.cuda()

        with Timer('Running model'):
            dev_loss = 0.0
            srl_predictions = []

            model.eval()
            with torch.no_grad():  # Eval don't need the grad
                matched_pred = sys_pred = gold_pred = 0
                matched_argu = sys_argu = gold_argu = 0
                matched_srl = sys_srl = gold_srl = 0
                t_pred_args_pairs = 0
                for i, batched_tensor in enumerate(test_data):
                    sent_ids, sent_lengths, word_indexes, head_indexes, char_indexes,\
                        predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens,\
                        gold_predicates, num_gold_predicates, target_predicates = batched_tensor

                    if args.gpu:
                        word_indexes, head_indexes, char_indexes,\
                            predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens, target_predicates = \
                            word_indexes.cuda(), head_indexes.cuda(), char_indexes.cuda(), predicate_indexes.cuda(), arg_starts.cuda(), \
                            arg_ends.cuda(), arg_labels.cuda(), srl_lens.cuda(), target_predicates.cuda()
                        # gold_predicates.cuda(), num_gold_predicates.cuda()
                    # print(i, "begin")
                    predicated_dict, loss = model.forward(
                        sent_lengths, word_indexes, head_indexes, char_indexes,
                        (predicate_indexes, arg_starts, arg_ends, arg_labels,
                         srl_lens), (gold_predicates, num_gold_predicates,
                                     target_predicates))
                    # print(i, "done")
                    dev_loss += float(loss)

                    num_pred_args_pairs = predicated_dict[
                        "batch_size"] * predicated_dict[
                            "max_num_preds"] * predicated_dict["max_num_args"]
                    t_pred_args_pairs += num_pred_args_pairs
                    print("num_sentence", predicated_dict["batch_size"],
                          "num_preds", predicated_dict["max_num_preds"],
                          "num_args", predicated_dict["max_num_args"],
                          num_pred_args_pairs)

                    matched_pred += predicated_dict["match_predicates"]
                    sys_pred += predicated_dict["sys_predicate"]
                    gold_pred += predicated_dict["gold_predicate"]
                    matched_argu += predicated_dict["matched_argu_num"]
                    sys_argu += predicated_dict["sys_argu_num"]
                    gold_argu += predicated_dict["gold_argu_num"]
                    matched_srl += predicated_dict["matched_srl_num"]
                    sys_srl += predicated_dict["sys_srl_num"]
                    gold_srl += predicated_dict["gold_srl_num"]

                    decoded_predictions = srl_decode(sent_lengths,
                                                     predicated_dict,
                                                     label_dict.idx2str,
                                                     config)
                    #
                    if "srl" in decoded_predictions:
                        srl_predictions.extend(decoded_predictions["srl"])
            try:
                pred_p = 1.0 * matched_pred / sys_pred
                pred_r = 1.0 * matched_pred / gold_pred
                pred_f = 2.0 * pred_p * pred_r / (pred_p + pred_r)
                print(matched_pred, sys_pred, gold_pred)
                print(pred_p, pred_r, pred_f)

                argu_p = 1.0 * matched_argu / sys_argu
                argu_r = 1.0 * matched_argu / gold_argu
                argu_f = 2.0 * argu_p * argu_r / (argu_p + argu_r)
                print(matched_argu, sys_argu, gold_argu)
                print(argu_p, argu_r, argu_f)

                srl_p = 1.0 * matched_srl / sys_srl
                srl_r = 1.0 * matched_srl / gold_srl
                srl_f = 2.0 * srl_p * srl_r / (srl_p + srl_r)
                print(matched_srl, sys_srl, gold_srl)
                print(srl_p, srl_r, srl_f)
            except Exception:
                print("pass because of zero division")
                # return
            print("total pred-argument-pairs", t_pred_args_pairs)

            sentences, gold_srl, gold_ner = zip(*eval_data)
            # Summarize results, evaluate entire dev set.
            precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp = \
                (compute_srl_f1(sentences, gold_srl, srl_predictions, args.gold, config.span_based, args.mate_tools_senses))
