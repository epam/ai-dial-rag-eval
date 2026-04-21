import math
from typing import List, Tuple

import pytest

from aidial_rag_eval.generation.inference import _aggregate_segment_inferences
from aidial_rag_eval.generation.types import ErrorInfo, InferenceInputs, InferenceScore


def _make_group_item(
    statements: List[str],
    inference: float,
    input_error: ErrorInfo | None = None,
    score_error: ErrorInfo | None = None,
) -> Tuple[InferenceInputs, InferenceScore]:
    inputs = InferenceInputs(
        hypothesis_id=0,
        premise="premise",
        statements=statements,
        document_name="",
        error=input_error,
    )
    score = InferenceScore(inference=inference, explanation="", error=score_error)
    return inputs, score


def _some_error() -> ErrorInfo:
    return ErrorInfo(error_repr="SomeError: msg", traceback="...")


def test_no_errors_weighted_mean():
    data = [
        _make_group_item(["s1", "s2"], inference=0.5),
        _make_group_item(["s3"], inference=1.0),
    ]
    inf, inf_min, inf_max = _aggregate_segment_inferences(data)
    assert inf == pytest.approx(2 / 3, abs=1e-4)
    assert inf_min == pytest.approx(2 / 3, abs=1e-4)
    assert inf_max == pytest.approx(2 / 3, abs=1e-4)


def test_no_errors_no_statements():
    data = [
        _make_group_item([], inference=0.0),
    ]
    inf, inf_min, inf_max = _aggregate_segment_inferences(data)
    assert math.isnan(inf)
    assert inf_min == pytest.approx(0.0, abs=1e-4)
    assert inf_max == pytest.approx(1.0, abs=1e-4)


def test_early_stage_error():
    error = _some_error()
    data = [
        _make_group_item([], inference=math.nan, input_error=error, score_error=error),
        _make_group_item([], inference=math.nan, input_error=error, score_error=error),
    ]
    inf, inf_min, inf_max = _aggregate_segment_inferences(data)
    assert math.isnan(inf)
    assert inf_min == pytest.approx(0.0, abs=1e-4)
    assert inf_max == pytest.approx(1.0, abs=1e-4)


def test_inference_stage_errors_partial():
    score_error = _some_error()
    data = [
        _make_group_item(["s1", "s2"], inference=0.5),
        _make_group_item(
            ["s3", "s4", "s5"], inference=math.nan, score_error=score_error
        ),
    ]
    inf, inf_min, inf_max = _aggregate_segment_inferences(data)
    assert math.isnan(inf)
    assert inf_min == pytest.approx(0.2, abs=1e-4)
    assert inf_max == pytest.approx(0.8, abs=1e-4)


def test_inference_stage_all_errors():
    score_error = _some_error()
    data = [
        _make_group_item(["s1", "s2"], inference=math.nan, score_error=score_error),
        _make_group_item(
            ["s3", "s4", "s5"], inference=math.nan, score_error=score_error
        ),
    ]
    inf, inf_min, inf_max = _aggregate_segment_inferences(data)
    assert math.isnan(inf)
    assert inf_min == pytest.approx(0.0, abs=1e-4)
    assert inf_max == pytest.approx(1.0, abs=1e-4)
