import numpy as np
import pytest

from rag_eval_metrics.metrics import (
    calculate_f1,
    calculate_mrr,
    calculate_precision,
    calculate_recall,
)


def test_precision():
    context_relevance = np.array([1, 0, 1, 0, 1])
    precision = calculate_precision(context_relevance)
    assert precision == 0.6


def test_recall():
    facts_ranks = np.array([1, 2, 3, -1])
    recall = calculate_recall(facts_ranks)
    assert recall == 0.75


def test_f1():
    precision = np.float64(0.6)
    recall = np.float64(0.75)
    f1 = calculate_f1(precision, recall)
    assert f1 == pytest.approx(2.0 / 3.0)


def test_mrr():
    facts_ranks = np.array([1, 2, 3, -1])
    mrr = calculate_mrr(facts_ranks)
    assert mrr == pytest.approx((1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0) / 4)
