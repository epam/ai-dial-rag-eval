import numpy as np
import pandas as pd
import pytest

from aidial_rag_eval.metrics import (
    calculate_f1,
    calculate_metrics,
    calculate_mrr,
    calculate_precision,
    calculate_recall,
)
from aidial_rag_eval.retrieval.types import FactMatchResult


def test_precision():
    context_relevance = np.array([1, 0, 1, 0, 1])
    precision = calculate_precision(context_relevance=context_relevance)
    assert precision == 0.6


def test_recall():
    facts_ranks = np.array([1, 2, 3, -1])
    recall = calculate_recall(facts_ranks=facts_ranks)
    assert recall == 0.75


def test_f1():
    precision = np.float64(0.6)
    recall = np.float64(0.75)
    f1 = calculate_f1(precision, recall)
    assert f1 == pytest.approx(2.0 / 3.0)


def test_mrr():
    facts_ranks = np.array([1, 2, 3, -1])
    mrr = calculate_mrr(facts_ranks=facts_ranks)
    assert mrr == pytest.approx((1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0) / 4)


def test_metrics():
    match_result = FactMatchResult(
        facts_ranks=np.array([1, 2, 3, -1]),
        context_relevance=np.array([1, 0, 1, 0, 1]),
        context_highlight=np.array(["", "", "", "", ""]),
    )
    metrics = calculate_metrics(match_result)
    assert metrics == {
        "recall": pytest.approx(0.75),
        "precision": pytest.approx(0.6),
        "f1": pytest.approx(2.0 / 3.0),
        "mrr": pytest.approx((1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0) / 4),
    }


def test_metrics_from_dataframe():
    data = pd.DataFrame(
        [
            {
                "facts_ranks": np.array([3, 2]),
                "context_relevance": np.array([0, 0, 1, 1]),
            },
            {
                "facts_ranks": np.array([1]),
                "context_relevance": np.array([1, 0, 0, 0, 0]),
            },
            {"facts_ranks": np.array([0, -1]), "context_relevance": np.array([1, 0])},
            {"facts_ranks": np.array([-1]), "context_relevance": np.array([0, 0])},
        ]
    )
    metrics = data.apply(calculate_metrics, axis=1, result_type="expand")

    pd.testing.assert_series_equal(
        metrics["recall"],
        pd.Series([1, 1, 0.5, 0], name="recall"),
    )

    pd.testing.assert_series_equal(
        metrics["precision"],
        pd.Series([0.5, 0.2, 0.5, 0], name="precision"),
    )

    pd.testing.assert_series_equal(
        metrics["f1"],
        pd.Series([2.0 / 3.0, 1.0 / 3.0, 0.5, 0], name="f1"),
    )

    pd.testing.assert_series_equal(
        metrics["mrr"],
        pd.Series([0.29166666, 0.5, 0.5, 0], name="mrr"),
    )
