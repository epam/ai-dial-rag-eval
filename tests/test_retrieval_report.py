from pathlib import Path

import fsspec
import pandas as pd

from aidial_rag_eval.dataframe.metrics import create_retrieval_metrics_report

TEST_DATA_PATH = Path(__file__).parent / "data"


# For s3: "dir::s3://<bucket>/"
test_fs, _ = fsspec.url_to_fs(f"dir::file://{TEST_DATA_PATH}/")


def test_data_from_fsspec():
    ground_truth = pd.read_parquet(
        "ground_truth_3.parquet",
        filesystem=test_fs,
        columns=["documents", "question", "facts"],
    )
    answers = pd.read_parquet(
        "answers_3.parquet",
        filesystem=test_fs,
        columns=["documents", "question", "context"],
    )

    metrics = create_retrieval_metrics_report(ground_truth, answers)
    expected_metrics = pd.DataFrame(
        {
            "recall": [0.5, 1.0, 0.0, 0.0, 0.0],
            "precision": [0.071428, 0.076923, 0.0, 0.0, 0.0],
            "f1": [0.125, 0.142857, 0.0, 0.0, 0.0],
            "mrr": [0.125, 0.125, 0.0, 0.0, 0.0],
        }
    )

    print(metrics)

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
