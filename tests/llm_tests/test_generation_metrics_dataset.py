from pathlib import Path

import pytest

from aidial_rag_eval.evaluate import evaluate
from aidial_rag_eval.metric_binds import (
    ANSWER_REFUSAL,
    ANSWER_TO_GROUND_TRUTH_INFERENCE,
    CONTEXT_TO_ANSWER_INFERENCE,
    GROUND_TRUTH_REFUSAL,
    GROUND_TRUTH_TO_ANSWER_INFERENCE,
)

TEST_DATA_PATH = Path(__file__).parent.parent / "data"


def test_inference_dataset(tmp_path, llm):
    ground_truth = (
        f"file:///{TEST_DATA_PATH}/ground_truth_inference_1.parquet.metadata.json"
    )
    answers = f"file:///{TEST_DATA_PATH}/answers_inference_1.parquet.metadata.json"

    dest = f"file:///{tmp_path}/metrics_dummy_inference.parquet"

    metrics = evaluate(
        str(ground_truth),
        str(answers),
        str(dest),
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

    sources_metadata_paths = [
        source.metadata_path for source in metrics.metadata.sources
    ]
    assert str(ground_truth) in sources_metadata_paths
    assert str(answers) in sources_metadata_paths

    assert "aidial-rag-eval" in metrics.metadata.tools

    print(metrics.metadata.metrics)
    assert metrics.metadata.metrics == {
        "recall": pytest.approx(0.666667, abs=1e-4),
        "precision": pytest.approx(0.666667, abs=1e-4),
        "mrr": pytest.approx(0.666667, abs=1e-4),
        "f1": pytest.approx(0.666667, abs=1e-4),
        "ctx_ans_inference": pytest.approx(0.333333, abs=1e-4),
        "ans_gt_inference": pytest.approx(0.333333, abs=1e-4),
        "gt_ans_inference": pytest.approx(0.333333, abs=1e-4),
        "answer_refusal": pytest.approx(0.333333, abs=1e-4),
        "ground_truth_refusal": pytest.approx(0.0, abs=1e-4),
        "mean_inference": pytest.approx(0.333333, abs=1e-4),
        "median_inference": pytest.approx(0.333333, abs=1e-4),
    }

    print(metrics.metadata.statistics)
    assert metrics.metadata.statistics == {
        "Ground truth size": 1,
        "Answers size": 3,
        "Evaluation data size": 3,
    }

    metrics_df = metrics.read_dataframe()
    assert set(metrics_df.columns.tolist()) == {
        "question",
        "context",
        "ground_truth_answer",
        "answer",
        "documents",
        "ctx_ans_inference",
        "ctx_ans_json",
        "ctx_ans_highlight",
        "ans_gt_inference",
        "ans_gt_json",
        "ans_gt_highlight",
        "gt_ans_inference",
        "gt_ans_json",
        "gt_ans_highlight",
        "answer_refusal",
        "ground_truth_refusal",
        "mean_inference",
        "median_inference",
        "facts",
        "facts_ranks",
        "context_relevance",
        "context_highlight",
        "recall",
        "precision",
        "mrr",
        "f1",
    }
