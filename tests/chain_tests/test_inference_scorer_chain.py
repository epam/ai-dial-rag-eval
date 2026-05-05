import json
import math

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
)
from aidial_rag_eval.generation.types import InferenceInputs
from tests.chain_tests.fake_models import FakeRecordingChatModel

# flake8: noqa: E501
EXPECTED_INFERENCE_PROMPT = """
Natural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
A hypothesis is a list of statements provided below.

The name of the document from which the premise was derived is also provided (if available).

A statement is considered an entailment if it is a paraphrase of information expressed in the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
A statement is considered neutral if the premise neither supports nor contradicts it, or if the statement contains information the premise does not address.

A statement can be entailed even if the premise contains more information than the statement covers \u2014 a statement that restates only part of the premise, or condenses it, is still entailment, as long as it does not introduce new information.

Important: Do not rely on factual world knowledge or logical inference chains to establish entailment \u2014 if a fact is not stated in the premise, it is not entailed. However, recognizing synonyms and paraphrases is not inference: it is identifying the same meaning expressed in different words, which is a core part of determining entailment.

For each statement:
Provide a brief short(1 sentence) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.

Format your response in JSON. You must return only JSON.

For example, given the following request:
{
  "document_name": "biology_article",
  "premise": "I am a biology graduate and I work at a tech company.",
  "statements": [
    "I am a graduate.",
    "I work at a hospital.",
    "I am employed at a tech firm."
  ]
}
the expected output is:
{
  "statement_inference": [
    {
      "explanation": "It is true that I am a graduate.",
      "tag": "ENT"
    },
    {
      "explanation": "Premise states I work at a tech company, not a hospital.",
      "tag": "CONT"
    },
    {
      "explanation": "Employed at a tech firm is a paraphrase of working at a tech company.",
      "tag": "ENT"
    }
  ]
}

Request:
{
  "document_name": "test_doc",
  "premise": "Water is wet.",
  "statements": [
    "Water is wet."
  ]
}"""


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
        responses=[json.dumps({"results": [{"tag": "ENT", "explanation": "test"}]})]
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
    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_json_missing_tag_key():
    fake_llm = FakeListChatModel(
        responses=[json.dumps({"results": [{"explanation": "value"}]})]
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


@pytest.mark.skip(reason="explanation key check is not implemented")
def test_json_missing_explanation_key():
    fake_llm = FakeListChatModel(responses=[json.dumps({"results": [{"tag": "ENT"}]})])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_output_count_mismatch():
    fake_llm = FakeListChatModel(
        responses=[json.dumps({"results": [{"tag": "ENT", "explanation": "test"}]})]
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1", "Statement2"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_empty_statements():
    fake_llm = FakeListChatModel(responses=["should not be called"])
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input([])]

    results = scorer.get_inference(inputs, show_progress_bar=False)

    assert results[0].inference == 0.0
    assert results[0].explanation == ""
    assert results[0].error is None


def test_invoke_raises_exception():
    fake_llm = FakeRecordingChatModel(
        responses=[], side_effect=Exception("LLM invoke failed")
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)
    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None


def test_prompt_contains_statements():
    statements = ["Water is wet."]
    fake_llm = FakeRecordingChatModel(
        responses=[
            json.dumps({"statement_inference": [{"tag": "ENT", "explanation": "test"}]})
        ]
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    scorer.get_inference([_create_inference_input(statements)], show_progress_bar=False)

    assert fake_llm.recorded_inputs[0][0].content == EXPECTED_INFERENCE_PROMPT
