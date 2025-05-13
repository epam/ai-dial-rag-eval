from pathlib import Path

import fsspec
import pandas as pd

from aidial_rag_eval.dataframe import create_rag_eval_metrics_report
from aidial_rag_eval.metric_binds import (
    ANSWER_REFUSAL,
    ANSWER_TO_GROUND_TRUTH_INFERENCE,
    CONTEXT_TO_ANSWER_INFERENCE,
    GROUND_TRUTH_REFUSAL,
    GROUND_TRUTH_TO_ANSWER_INFERENCE,
)

TEST_DATA_PATH = Path(__file__).parent.parent / "data"


# For s3: "dir::s3://<bucket>/"
test_fs, _ = fsspec.url_to_fs(f"dir::file://{TEST_DATA_PATH}/")


def test_data_inference_from_fsspec(llm):
    ground_truth = pd.read_parquet(
        "ground_truth_inference_1.parquet", filesystem=test_fs
    )
    answers = pd.read_parquet(
        "answers_inference_1.parquet",
        filesystem=test_fs,
    )

    metrics = create_rag_eval_metrics_report(
        ground_truth,
        answers,
        llm=llm,
        metric_binds=[
            CONTEXT_TO_ANSWER_INFERENCE,
            ANSWER_TO_GROUND_TRUTH_INFERENCE,
            GROUND_TRUTH_TO_ANSWER_INFERENCE,
            ANSWER_REFUSAL,
            GROUND_TRUTH_REFUSAL,
        ],
        show_progress_bar=False,
    )

    expected_metrics = pd.DataFrame(
        {
            "recall": [1.0, 1.0, 0.0],
            "precision": [1.0, 1.0, 0.0],
            "f1": [1.0, 1.0, 0.0],
            "mrr": [1.0, 1.0, 0.0],
            "ctx_ans_inference": [0.0, 1.0, 0.0],
            "ans_gt_inference": [0.0, 1.0, 0.0],
            "gt_ans_inference": [0.0, 1.0, 0.0],
            "mean_inference": [0.0, 1.0, 0.0],
            "median_inference": [0.0, 1.0, 0.0],
            "answer_refusal": [0.0, 0.0, 1.0],
            "ground_truth_refusal": [0.0, 0.0, 0.0],
        }
    )

    print(metrics)
    print(expected_metrics)
    for columns in expected_metrics.columns:
        pd.testing.assert_series_equal(
            metrics[columns],
            expected_metrics[columns],
            atol=1e-4,
        )
