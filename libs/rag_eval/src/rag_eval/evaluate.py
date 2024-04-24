import pandas as pd
from rag_eval_metrics.facts.citation import CitationMatcher
from rag_eval_metrics.metrics import (
    calculate_f1,
    calculate_mrr,
    calculate_precision,
    calculate_recall,
)


def evaluate_retrieval(row):
    ground_truth_facts = row["ground_truth_facts"]
    context = row["context"]
    facts_ranks, context_relevance = CitationMatcher.match_facts(
        ground_truth_facts, context
    )

    recall = calculate_recall(facts_ranks)
    precision = calculate_precision(context_relevance)
    f1 = calculate_f1(recall, precision)

    eval_result = pd.Series(
        {
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "MRR": calculate_mrr(facts_ranks),
            "len_facts": len(ground_truth_facts),
            "len_context": len(context),
        }
    )

    return eval_result


def evaluation_report(ground_truth, answers, name="evaluation"):
    eval_data = pd.merge(ground_truth, answers, on="question")
    print(f"Ground truth: {len(ground_truth)}")
    print(f"Answers: {len(answers)}")
    print(f"Eval data dial-rag: {len(eval_data)}")
    metrics = eval_data.apply(evaluate_retrieval, axis=1)
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
