import sys
# from eval_orl_conll_file import load_eval_data, analyze_error_prediction_matrix
from eval_orl_e2e_json_file import load_eval_data, analyze_error_prediction_matrix


def average_fscore(all_metrics):
    zipped_metrics = list(zip(*all_metrics))

    def avg(metrics):
        return sum([item.f for item in metrics]) / len(metrics)

    print('='*10, "Binary F1", '='*10)
    print("Agent", avg(zipped_metrics[0]))
    print("Target", avg(zipped_metrics[1]))
    print("Agent", avg(zipped_metrics[2]))

    print('='*10, "Proportional F1", '='*10)
    print("Agent", avg(zipped_metrics[3]))
    print("Target", avg(zipped_metrics[4]))
    print("Agent", avg(zipped_metrics[5]))

    print('='*10, "Exact F1", '='*10)
    print("Agent", avg(zipped_metrics[6]))
    print("Target", avg(zipped_metrics[7]))
    print("Agent", avg(zipped_metrics[8]))

    print('=' * 10, "Expression F1", '=' * 10)
    print("Binary", avg(zipped_metrics[9]))
    print("Proportional", avg(zipped_metrics[10]))
    print("Exact", avg(zipped_metrics[11]))


if __name__ == "__main__":
    averaged_metric = []
    assert len(sys.argv[1:]) == 5
    for file_path in sys.argv[1:]:
        result = load_eval_data(file_path)

        x = analyze_error_prediction_matrix(result)
        (agent_binary, target_binary, all_binary), \
        (agent_proportional, target_proportional, all_proportional), \
        (agent_exact, target_exact, all_exact), \
        (exp_binary, exp_proportional, exp_exact) = x

        averaged_metric.append([agent_binary, target_binary, all_binary,
                                agent_proportional, target_proportional, all_proportional,
                                agent_exact, target_exact, all_exact,
                                exp_binary, exp_proportional, exp_exact])
    average_fscore(averaged_metric)

