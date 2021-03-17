# Inference functions for the SRL model.
import numpy as np


# TODO: Pass arg
def decode_spans(span_starts, span_ends, span_scores, labels_inv):
    """
  Args:
    span_starts: [num_candidates,]
    span_scores: [num_candidates, num_labels]
  """
    pred_spans = []
    span_labels = np.argmax(span_scores, axis=1)  # [num_candidates]
    spans_list = zip(span_starts, span_ends, span_labels, span_scores)
    spans_list = sorted(spans_list, key=lambda x: x[3][x[2]], reverse=True)
    predicted_spans = {}
    for start, end, label, _ in spans_list:
        # Skip invalid span.
        if label == 0 or (start, end) in predicted_spans:
            continue
        pred_spans.append((start, end, labels_inv[label]))
        predicted_spans[(start, end)] = label
    return pred_spans


def greedy_decode(predict_dict, srl_labels_inv):
    """Greedy decoding for SRL predicate-argument structures.

  Args:
    predict_dict: Dictionary of name to numpy arrays.
    srl_labels_inv: SRL label id to string name.
    suppress_overlap: Whether to greedily suppress overlapping arguments for the same predicate.

  Returns:
    A dictionary from predicate ids to lists of argument spans.
  """
    arg_starts = predict_dict["arg_starts"]
    arg_ends = predict_dict["arg_ends"]
    predicates = predict_dict["predicates"]
    arg_labels = predict_dict["arg_labels"]
    scores = predict_dict["srl_scores"]

    num_suppressed_args = 0

    # Map from predicates to a list of labeled spans.
    pred_to_args = {}
    if len(arg_ends) > 0 and len(predicates) > 0:
        max_len = max(np.max(arg_ends), np.max(predicates)) + 1
    else:
        max_len = 1

    for j, pred_id in enumerate(predicates):
        args_list = []
        for i, (arg_start, arg_end) in enumerate(zip(arg_starts, arg_ends)):
            # If label is not null.
            if arg_labels[i][j] == 0:
                continue
            label = srl_labels_inv[arg_labels[i][j]]
            # if label not in ["V", "C-V"]:
            args_list.append((arg_start, arg_end, label, scores[i][j][arg_labels[i][j]]))

        # Sort arguments by highest score first.
        args_list = sorted(args_list, key=lambda x: x[3], reverse=True)
        new_args_list = []

        flags = [False for _ in range(max_len)]
        # Predicate will not overlap with arguments either.
        flags[pred_id] = True

        for (arg_start, arg_end, label, score) in args_list:
            # If none of the tokens has been covered:
            if not max(flags[arg_start:arg_end + 1]):
                new_args_list.append((arg_start, arg_end, label))
                for k in range(arg_start, arg_end + 1):
                    flags[k] = True

        # Only add predicate if it has any argument.
        if new_args_list:
            pred_to_args[pred_id] = new_args_list

        num_suppressed_args += len(args_list) - len(new_args_list)

    return pred_to_args, num_suppressed_args


_CORE_ARGS = {"ARG0": 1, "ARG1": 2, "ARG2": 4, "ARG3": 8, "ARG4": 16, "ARG5": 32, "ARGA": 64,
              "A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    return predicted_clusters, mention_to_predicted


def _decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id):
    labels = np.argmax(scores, axis=1)
    spans = []
    for i, (start, end, label) in enumerate(zip(starts, ends, labels)):
        if label <= 0:
            continue
        label_str = labels_inv[label]
        if pred_id is not None and label_str == "V":
            continue
        spans.append((start, end, label_str, scores[i][label]))
    spans = sorted(spans, key=lambda x: x[3], reverse=True)
    flags = np.zeros([max_len], dtype=bool)
    if pred_id is not None:
        flags[pred_id] = True
    new_spans = []
    for start, end, label_str, score in spans:
        if not max(flags[start:end + 1]):
            new_spans.append((start, end, label_str))  # , score))
            for k in range(start, end + 1):
                flags[k] = True
    return new_spans


def _dp_decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, d_s, d_e, u_constraint=False):
    num_roles = scores.size()[1]  # [num_arg, num_roles]
    starts, ends, scores = starts.numpy().astype(np.int64), ends.numpy().astype(np.int64), scores.data.cpu().numpy()

    labels = np.argmax(scores, axis=1).astype(np.int64)
    spans = zip(starts, ends, range(len(starts)))
    spans = sorted(spans, key=lambda x: (x[0], x[1]))  # sort according to the span start index

    if u_constraint:
        f = np.zeros([max_len + 1, 128], dtype=float) - 0.1
    else:  # This one
        f = np.zeros([max_len + 1, 1], dtype=float) - 0.1

    f[0, 0] = 0
    states = {0: set([0])}  # A dictionary from id to list of binary core-arg states.
    pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs)
    best_state = [(0, 0)]

    def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
        if f[t0][rs0] + delta > f[t1][rs1]:
            f[t1][rs1] = f[t0][rs0] + delta
            if t1 not in states:
                states[t1] = set()
            states[t1].update([rs1])
            pointers[(t1, rs1)] = (arg_id, role, t0, rs0)  # the pointers store
            if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
                best_state[0] = (t1, rs1)

    results = []
    for start, end, i in spans:  # [arg_start, arg_end, arg_span_id]
        assert scores[i][0] == 0  # kiro, original [0]
        # The extra dummy score should be same for all states, so we can safely skip arguments overlap
        # with the predicate.
        flag = False
        for idx in range(start, end + 1):
            if d_s <= idx <= d_e:  # skip the span contains the predicate
                flag = True
                break
        if flag is True:
            continue
        r0 = labels[i]  # Locally best role assignment.
        # Strictly better to incorporate a dummy span if it has the highest local score.
        if r0 == 0:  # labels_inv[r0] == "O"
            continue
        r0_str = labels_inv[r0]
        # results.append((start, end, r0_str))
        # continue
        # Enumerate explored states.
        t_states = [t for t in states.keys() if t <= start]  # collect the state which is before the current span
        for t in t_states:  # for each state
            role_states = states[t]
            # Update states if best role is not a core arg.
            if not u_constraint or r0_str not in _CORE_ARGS:  # True; this one
                for rs in role_states:  # the set type in the value in the state dict
                    _update_state(t, rs, end + 1, rs, scores[i][r0], i, r0)  # update the state
            else:
                for rs in role_states:
                    for r in range(1, num_roles):
                        if scores[i][r] > 0:
                            r_str = labels_inv[r]
                            core_state = _CORE_ARGS.get(r_str, 0)
                            # print start, end, i, r_str, core_state, rs
                            if core_state & rs == 0:
                                _update_state(t, rs, end + 1, rs | core_state, scores[i][r], i, r)
    # return results
    # Backtrack to decode.
    new_spans = []
    t, rs = best_state[0]
    while (t, rs) in pointers:
        i, r, t0, rs0 = pointers[(t, rs)]
        new_spans.append((starts[i], ends[i], labels_inv[r]))
        t = t0
        rs = rs0
    return new_spans[::-1]


def srl_decode(sentence_lengths, predict_dict, srl_labels_inv, config, cons_idx2label=None):  # decode the predictions.
    predictions = {}

    # Decode sentence-level tasks.
    num_sentences = len(sentence_lengths)
    predictions["srl"] = [{} for _ in range(num_sentences)]
    predictions['dse'] = [[] for _ in range(num_sentences)]
    predictions["arg"] = [[] for _ in range(num_sentences)]
    # Sentence-level predictions.
    for i in range(num_sentences):  # for each sentences
        if "srl" in predictions:
            num_args = predict_dict["num_args"][i]  # the number of the candidate argument spans
            num_dse = predict_dict["num_dse"][i]  # the number of the candidate predicates
            # print(num_args, num_preds)
            # for each predicate id, exec the decode process
            for j, (dse_start, dse_end) in enumerate(zip(predict_dict["dse_starts"][i][:num_dse],
                                            predict_dict["dse_ends"][i][:num_dse])):
                # sorted arg_starts and arg_ends and srl_scores ? should be??? enforce_srl_constraint = False
                # print(predict_dict["srl_scores"].size(), i, num_args, j)
                dse_start, dse_end = int(dse_start), int(dse_end)
                predictions["dse"][i].append((dse_start, dse_end))
                if int(num_args) == 0 or int(num_dse) == 0:
                    pass
                arg_spans = _dp_decode_non_overlapping_spans(
                    predict_dict["arg_starts"][i][:num_args],
                    predict_dict["arg_ends"][i][:num_args],
                    predict_dict["srl_scores"][i, :num_args, j, :],
                    sentence_lengths[i], srl_labels_inv, dse_start, dse_end, config.enforce_srl_constraint)
                # To avoid warnings in the eval script.
                if config.use_gold_predicates:  # false
                    pass
                    # arg_spans.append((pred_id, pred_id, "V"))
                    # print(i, arg_spans)
                predictions["srl"][i][(dse_start, dse_end)] = []
                if arg_spans:
                    for sys_span in sorted(arg_spans, key=lambda x: (x[0], x[1])):
                        s, e, label = sys_span
                        predictions["srl"][i][(dse_start, dse_end)].append((s, e, label))

    if config.analyze:
        for i in range(num_sentences):  # for each sentences
            if "arg" in predictions:
                num_sys_args = predict_dict["sys_arg_num"][i]
                for j, (arg_start, argu_end) in enumerate(zip(
                        predict_dict["sys_arg_starts"][i][:num_sys_args],
                        predict_dict["sys_arg_ends"][i][:num_sys_args])):
                    predictions["arg"][i].append([int(arg_start), int(argu_end)])

        predictions["cons"] = [[] for _ in range(num_sentences)]
        for i in range(num_sentences):  # for each sentences
            num_args = predict_dict["sys_cons_nums"][i]
            scores = predict_dict["sys_cons_scores"][i].data.cpu().numpy()
            label_idx = np.argmax(scores, axis=1).astype(np.int64)
            for j, (start, end) in enumerate(zip(predict_dict["sys_cons_starts"][i][:num_args],
                                            predict_dict["sys_cons_ends"][i][:num_args])):
                start, end = int(start), int(end)
                # print(scores, label_idx)
                predictions["cons"][i].append([start, end, cons_idx2label[label_idx[j]]])

        predictions["argu_cons"] = [[] for _ in range(num_sentences)]
        for i in range(num_sentences):  # for each sentences
            num_args = predict_dict["sys_argu_nums"][i]
            scores = predict_dict["sys_argu_scores"][i].data.cpu().numpy()
            label_idx = np.argmax(scores, axis=1).astype(np.int64)
            for j, (start, end) in enumerate(zip(predict_dict["sys_argu_starts"][i][:num_args],
                                            predict_dict["sys_argu_ends"][i][:num_args])):
                start, end = int(start), int(end)
                # print(scores, label_idx)
                predictions["argu_cons"][i].append([start, end, cons_idx2label[label_idx[j]]])

    return predictions
