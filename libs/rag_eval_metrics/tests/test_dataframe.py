from pathlib import Path

import fsspec
import pandas as pd

from rag_eval_metrics.dataframe.match_facts import match_facts_dataframe
from rag_eval_metrics.dataframe.metrics import calculate_metrics

TEST_DATA_PATH = Path(__file__).parent / "data"


# For s3: "dir::s3://<bucket>/"
test_fs, _ = fsspec.url_to_fs(f"dir::file://{TEST_DATA_PATH}/")


def test_data_from_fsspec():
    ground_truth = pd.read_parquet(
        "ground_truth_1.parquet",
        filesystem=test_fs,
        columns=["question", "facts"],
    )
    answers = pd.read_parquet(
        "answers_1.parquet",
        filesystem=test_fs,
        columns=["question", "context"],
    )

    matched_result = match_facts_dataframe(ground_truth, answers)

    expected_matched = pd.DataFrame(
        {
            "facts_ranks": [
                [-1, 3],
                [7],
                [-1, -1],
                [-1],
                [-1],
            ],
            "context_relevance": [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }
    )
    pd.testing.assert_series_equal(
        matched_result.facts_ranks,
        expected_matched.facts_ranks,
    )
    pd.testing.assert_series_equal(
        matched_result.context_relevance,
        expected_matched.context_relevance,
    )

    metrics = calculate_metrics(matched_result)
    expected_metrics = pd.DataFrame(
        {
            "recall": [0.5, 1.0, 0.0, 0.0, 0.0],
            "precision": [0.071428, 0.076923, 0.0, 0.0, 0.0],
            "f1": [0.125, 0.142857, 0.0, 0.0, 0.0],
            "mrr": [0.125, 0.125, 0.0, 0.0, 0.0],
        }
    )

    print(metrics)
    print(expected_metrics)

    pd.testing.assert_series_equal(
        metrics.recall,
        expected_metrics.recall,
    )
    pd.testing.assert_series_equal(
        metrics.precision,
        expected_metrics.precision,
    )
    pd.testing.assert_series_equal(
        metrics.f1,
        expected_metrics.f1,
    )
    pd.testing.assert_series_equal(
        metrics.mrr,
        expected_metrics.mrr,
    )
