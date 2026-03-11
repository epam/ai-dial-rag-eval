from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


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

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

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
    fake_llm = FakeListChatModel(responses=[""])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    with patch.object(
        FakeListChatModel, "batch", side_effect=Exception("LLM invoke failed")
    ):
        result = extractor.extract(
            [
                SegmentedText(
                    hypothesis_segments, [" "] * (len(hypothesis_segments) - 1)
                )
            ],
            show_progress_bar=False,
        )[0]
        assert isinstance(result, ErrorInfo)
