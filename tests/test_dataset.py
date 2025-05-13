from pathlib import Path

import pytest

from aidial_rag_eval.evaluate import evaluate

TEST_DATA_PATH = f"{Path(__file__).parent}/data"


def test_dataset(tmp_path):
    ground_truth = f"file:///{TEST_DATA_PATH}/ground_truth.parquet.metadata.json"
    answers = f"file:///{TEST_DATA_PATH}/answers.parquet.metadata.json"

    dest = f"file:///{tmp_path}/metrics.parquet"

    metrics = evaluate(str(ground_truth), str(answers), str(dest))

    sources_metadata_paths = [
        source.metadata_path for source in metrics.metadata.sources
    ]
    assert str(ground_truth) in sources_metadata_paths
    assert str(answers) in sources_metadata_paths

    assert "aidial-rag-eval" in metrics.metadata.tools

    print(metrics.metadata.metrics)
    assert metrics.metadata.metrics == {
        "recall": pytest.approx(0.3, abs=1e-4),
        "precision": pytest.approx(0.0297, abs=1e-4),
        "mrr": pytest.approx(0.05, abs=1e-4),
        "f1": pytest.approx(0.0535, abs=1e-4),
    }

    print(metrics.metadata.statistics)
    assert metrics.metadata.statistics == {
        "Ground truth size": 5,
        "Answers size": 5,
        "Evaluation data size": 5,
    }

    metrics_df = metrics.read_dataframe()
    assert set(metrics_df.columns.tolist()) == {
        "documents",
        "question",
        "facts",
        "context",
        "facts_ranks",
        "context_relevance",
        "context_highlight",
        "recall",
        "precision",
        "mrr",
        "f1",
        "answer",
        "ground_truth_answer",
    }
