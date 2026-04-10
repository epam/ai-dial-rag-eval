# flake8: noqa: E501
import math

from aidial_rag_eval.generation.models.inference_scorers.inference_template import (
    StatementInference,
    StatementInferenceOutput,
)
from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
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


EXPECTED_INFERENCE_PROMPT = (
    "\nNatural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.\n"
    "A hypothesis is a list of statements provided below.\n"
    "\n"
    "The name of the document from which the premise was derived is also provided (if available).\n"
    "\n"
    "A statement is considered an entailment if it is a paraphrase of information expressed in the premise.\n"
    "A statement is considered a contradiction if it is logically inconsistent with the premise.\n"
    "A statement is considered neutral if the premise neither supports nor contradicts it, or if the statement contains information the premise does not address.\n"
    "\n"
    "A statement can be entailed even if the premise contains more information than the statement covers \u2014 a statement that restates only part of the premise, or condenses it, is still entailment, as long as it does not introduce new information.\n"
    "\n"
    "Important: Do not rely on factual world knowledge or logical inference chains to establish entailment \u2014 if a fact is not stated in the premise, it is not entailed. However, recognizing synonyms and paraphrases is not inference: it is identifying the same meaning expressed in different words, which is a core part of determining entailment.\n"
    "\n"
    "For each statement:\n"
    "Provide a brief short(1 sentences) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.\n"
    'Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.\n'
    "\n"
    "For example, given the following request:\n"
    "{\n"
    '  "document_name": "biology_article",\n'
    '  "premise": "I am a biology graduate and I work at a tech company.",\n'
    '  "statements": [\n'
    '    "I am a graduate.",\n'
    '    "I work at a hospital.",\n'
    '    "I am employed at a tech firm."\n'
    "  ]\n"
    "}\n"
    "the expected output is:\n"
    "{\n"
    '  "statement_inference": [\n'
    "    {\n"
    '      "explanation": "It is true that I am a graduate.",\n'
    '      "tag": "ENT"\n'
    "    },\n"
    "    {\n"
    '      "explanation": "Premise states I work at a tech company, not a hospital.",\n'
    '      "tag": "CONT"\n'
    "    },\n"
    "    {\n"
    '      "explanation": "Employed at a tech firm is a paraphrase of working at a tech company.",\n'
    '      "tag": "ENT"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Request:\n"
    "{\n"
    '  "document_name": "test_doc",\n'
    '  "premise": "Water is wet.",\n'
    '  "statements": [\n'
    '    "Water is wet."\n'
    "  ]\n"
    "}\n"
    "\n"
    "Note: the example output above illustrates the expected data structure. When using function calling, return the data via a tool call with the same structure."
    "\n"
    "IMPORTANT: Complete this entire task in a SINGLE response. Call the tool EXACTLY ONCE with ALL results in that one call."
)


def test_valid_json_response():
    fake_llm = FakeStructuredChatModel(
        responses=[
            (
                EXPECTED_INFERENCE_PROMPT,
                StatementInferenceOutput(
                    statement_inference=[
                        StatementInference(explanation="test", tag="ENT")
                    ]
                ),
            )
        ]
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


def test_output_count_mismatch():
    fake_llm = FakeStructuredChatModel(
        responses=[
            StatementInferenceOutput(
                statement_inference=[StatementInference(explanation="test", tag="ENT")]
            )
        ]
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
        responses=[
            StatementInferenceOutput(
                statement_inference=[StatementInference(explanation="test", tag="ENT")]
            )
        ]
    )
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    scorer.get_inference([_create_inference_input(statements)], show_progress_bar=False)


def test_invoke_raises_exception():
    fake_llm = FakeStructuredChatModel(side_effect=Exception("LLM invoke failed"))
    scorer = LLMInferenceScorer(model=fake_llm, max_concurrency=1)

    inputs = [_create_inference_input(["Statement1"])]

    results = scorer.get_inference(inputs, show_progress_bar=False)
    assert math.isnan(results[0].inference)
    assert results[0].explanation == ""
    assert results[0].error is not None
