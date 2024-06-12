from typing import Union

import pandas as pd

from aidial_rag_eval.dataframe.match_facts import (
    ANSWERS_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    match_facts_dataframe,
)
from aidial_rag_eval.dataframe.metrics import calculate_metrics
from aidial_rag_eval.dataset import Dataset, source_dataset
from aidial_rag_eval.utils import get_tools_versions


def evaluate(
    ground_truth: Union[str, Dataset], answers: Union[str, Dataset], dest: str
) -> Dataset:
    ground_truth_dataset = source_dataset(ground_truth)
    answers_dataset = source_dataset(answers)

    ground_truth_df = ground_truth_dataset.read_dataframe(columns=GROUND_TRUTH_COLUMNS)
    answers_df = answers_dataset.read_dataframe(columns=ANSWERS_COLUMNS)

    matched_result_df = match_facts_dataframe(ground_truth_df, answers_df)
    metrics_df = calculate_metrics(matched_result_df)
    aggregated_metrics = metrics_df.mean(numeric_only=True)
    assert isinstance(aggregated_metrics, pd.Series)

    metrics = Dataset.write_dataframe(
        metrics_df,
        dest,
        sources=[ground_truth_dataset, answers_dataset],
        tools=get_tools_versions(),
        metrics=aggregated_metrics.to_dict(),
        statistics={
            "Ground truth size": len(ground_truth_df),
            "Answers size": len(answers_df),
            "Evaluation data size": len(matched_result_df),
        },
    )
    return metrics
