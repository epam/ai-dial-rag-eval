import pandas as pd
from rag_eval_metrics.dataframe.match_facts import match_facts_dataframe
from rag_eval_metrics.dataframe.metrics import calculate_metrics


def evaluation_report(ground_truth, answers, name="evaluation"):
    print(f"Ground truth: {len(ground_truth)}")
    print(f"Answers: {len(answers)}")

    ground_truth_renames = ground_truth.rename(columns={"ground_truth_facts": "facts"})
    eval_data = match_facts_dataframe(ground_truth_renames, answers)
    print(f"Eval data dial-rag: {len(eval_data)}")

    metrics = calculate_metrics(eval_data)
    metrics["len_facts"] = eval_data.facts_ranks.apply(len)
    metrics["len_context"] = eval_data.context_relevance.apply(len)

    metrics_aggregated = pd.DataFrame(
        pd.concat(
            [
                metrics.mean(),
                pd.Series(
                    {
                        "ground truth size": len(ground_truth),
                        "answers size": len(answers),
                        "evaluation data size": len(eval_data),
                    }
                ),
            ]
        ),
        columns=[name],
    )

    eval_data_with_metrics = pd.concat([eval_data, metrics], axis=1)
    return metrics_aggregated, eval_data_with_metrics
