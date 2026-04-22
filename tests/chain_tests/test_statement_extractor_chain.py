from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeRecordingChatModel

# flake8: noqa: E501
EXPECTED_STATEMENT_PROMPT = """
Break down each hypothesis into statements, if hypothesis is complex. Else if the hypothesis is already a single statement, return it unchanged as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

When a hypothesis contains an enumeration or list, split it so that each item in the list becomes a separate statement. Preserve the relationship from the parent clause in each statement.

The output list must contain exactly as many items as there are input hypotheses, in the same order: the first item corresponds to the first hypothesis, the second item to the second hypothesis, and so on.

Examples:
Input hypotheses (JSON array of strings, one hypothesis per element):
[
  "The sky is blue and the grass is green.",
  "Water boils at 100 degrees Celsius.",
  "The company has offices in Paris, London, and Berlin."
]

Expected output:
{
  "hypothesis_statements": [
    {
      "statements": [
        "The sky is blue.",
        "The grass is green."
      ]
    },
    {
      "statements": [
        "Water boils at 100 degrees Celsius."
      ]
    },
    {
      "statements": [
        "The company has an office in Paris.",
        "The company has an office in London.",
        "The company has an office in Berlin."
      ]
    }
  ]
}

Request:
Input hypotheses (JSON array of strings, one hypothesis per element):
[
  "hypothesis_segment1",
  "hypothesis_segment2"
]"""


def test_valid_json_response():
    fake_llm = FakeListChatModel(
        responses=[
            """
            {
                "hypothesis_statements":
                    [
                        {
                            "statements": ["statement11"]
                        },
                        {
                            "statements": ["statement21"]
                        }
                    ]
            }"""
        ]
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]

    assert not isinstance(result, ErrorInfo)
    assert result == [["statement11"], ["statement21"]]


def test_invalid_json_response():
    fake_llm = FakeListChatModel(responses=["not valid json"])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_json_wrong_structure():
    fake_llm = FakeListChatModel(responses=['{"wrong_key": "not a list"}'])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_statement_count_mismatch():
    fake_llm = FakeListChatModel(
        responses=['{"hypothesis_statements": [{"statements": ["statement1"]}]}']
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_invoke_raises_exception():
    fake_llm = FakeRecordingChatModel(
        responses=[], side_effect=Exception("LLM invoke failed")
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_prompt_contains_hypotheses():
    segments = ["hypothesis_segment1", "hypothesis_segment2"]
    fake_llm = FakeRecordingChatModel(
        responses=[
            '{"hypothesis_statements": [{"statements": ["s1"]}, {"statements": ["s2"]}]}'
        ]
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    extractor.extract(
        [SegmentedText(segments, [" "] * (len(segments) - 1))],
        show_progress_bar=False,
    )

    assert fake_llm.recorded_inputs[0][0].content == EXPECTED_STATEMENT_PROMPT
