from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
)
from aidial_rag_eval.generation.types import InferenceInputs


def _create_inference_input(
    statements: list[str], premise: str = "Water is wet."
) -> InferenceInputs:
    return InferenceInputs(
        hypothesis_id=0,
        premise=premise,
        statements=statements,
        document_name="test_doc",
    )


def test_valid_json_response():
    fake_llm = FakeListChatModel(
        responses=['{"results": [{"tag": "ENT", "explanation": "test"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Water is wet."])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert len(results) == 1
    assert results[0].inference == 1.0
    assert (
        results[0].explanation
        == '[{"tag": "ENT", "explanation": "test", "statement": "Water is wet."}]'
    )


def test_invalid_json_response():
    fake_llm = FakeListChatModel(responses=["not valid json"])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert len(results) == 1
    assert results[0].inference == 0.0
    assert results[0].explanation == ""


def test_json_missing_tag_key():
    fake_llm = FakeListChatModel(responses=['{"results": [{"explanation": "value"}]}'])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""


@pytest.mark.skip(reason="explanation key check is not implemented")
def test_json_missing_explanation_key():
    fake_llm = FakeListChatModel(responses=['{"results": [{"tag": "ENT"}]}'])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""


def test_output_count_mismatch():
    fake_llm = FakeListChatModel(
        responses=['{"results": [{"tag": "ENT", "explanation": "test"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1", "Statement2"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""


def test_empty_statements():
    fake_llm = FakeListChatModel(responses=["should not be called"])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input([])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""


def test_invoke_raises_exception():
    fake_llm = FakeListChatModel(responses=[""])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    with patch.object(
        FakeListChatModel, "invoke", side_effect=Exception("LLM invoke failed")
    ):
        try:
            scorer.get_inference(inputs, show_progress_bar=False)
            raise AssertionError("Expected exception was not raised")
        except Exception as e:
            assert str(e) == "LLM invoke failed"
