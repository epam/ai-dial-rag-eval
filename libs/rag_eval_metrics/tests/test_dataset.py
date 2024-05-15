from pathlib import Path

from rag_eval_metrics.evaluate import evaluate

TEST_DATA_PATH = f"{Path(__file__).parent}/data"


def test_dataset(tmp_path):
    ground_truth = f"file:///{TEST_DATA_PATH}/ground_truth_2.parquet.metadata.json"
    answers = f"file:///{TEST_DATA_PATH}/answers_2.parquet.metadata.json"

    dest = f"file:///{tmp_path}/metrics.parquet"

    metrics = evaluate(str(ground_truth), str(answers), str(dest))

    sources_metadata_paths = [
        source.metadata_path for source in metrics.metadata.sources
    ]
    assert str(ground_truth) in sources_metadata_paths
    assert str(answers) in sources_metadata_paths

    assert "rag-eval-metrics" in metrics.metadata.tools

    metrics_df = metrics.read_dataframe()
    assert metrics_df.columns.tolist() == [
        "documents",
        "question",
        "facts",
        "context",
        "facts_ranks",
        "context_relevance",
        "recall",
        "precision",
        "mrr",
        "f1",
    ]
