import math
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.inference_scorers.inference_template import (
    inference_prompt,
)
from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
    _make_inference_prompt_input,
)
from aidial_rag_eval.generation.types import InferenceInputs
from tests.chain_tests.fake_models import FakeStructuredChatModel


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
    fake_llm = FakeStructuredChatModel(
        responses=['{"statement_inference": [{"explanation": "test", "tag": "ENT"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Water is wet."])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert len(results) == 1
    assert results[0].inference == 1.0
    assert (
        results[0].explanation
        == '[{"explanation": "test", "tag": "ENT", "statement": "Water is wet."}]'
    )


def test_invalid_json_response():
    fake_llm = FakeStructuredChatModel(responses=["not valid json"])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert len(results) == 1
    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_json_missing_tag_key():
    fake_llm = FakeStructuredChatModel(
        responses=['{"statement_inference": [{"explanation": "value"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_json_missing_explanation_key():
    fake_llm = FakeStructuredChatModel(
        responses=['{"statement_inference": [{"tag": "ENT"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_output_count_mismatch():
    fake_llm = FakeStructuredChatModel(
        responses=['{"statement_inference": [{"explanation": "test", "tag": "ENT"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1", "Statement2"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_empty_statements():
    fake_llm = FakeStructuredChatModel(responses=[])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input([])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""
    assert results[0].error is None


def test_prompt_contains_statements():
    statements = ["Water is wet."]
    fake_llm = FakeStructuredChatModel(
        responses=['{"statement_inference": [{"explanation": "test", "tag": "ENT"}]}']
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    scorer.get_inference([_create_inference_input(statements)], show_progress_bar=False)

    assert len(fake_llm.received_messages) == 1
    expected_prompt = inference_prompt.format(
        **_make_inference_prompt_input(
            premise="Water is wet.",
            statements=statements,
            document="test_doc",
        )
    )
    assert fake_llm.received_messages[0][-1].content == expected_prompt


def test_invoke_raises_exception():
    fake_llm = FakeStructuredChatModel(responses=[""])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    with patch.object(
        FakeListChatModel, "batch", side_effect=Exception("LLM invoke failed")
    ):
        results = scorer.get_inference(inputs, show_progress_bar=False)
        assert math.isnan(results[0].inference)
        assert results[0].explanation == ""
        assert results[0].error is not None