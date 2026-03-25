from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
    _make_statement_prompt_input,
)
from aidial_rag_eval.generation.models.statement_extractor.statement_extractor_template import (
    get_statement_prompt,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeStructuredChatModel


def test_valid_json_response():
    fake_llm = FakeStructuredChatModel(
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
    fake_llm = FakeStructuredChatModel(responses=["not valid json"])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_json_wrong_structure():
    fake_llm = FakeStructuredChatModel(responses=['{"wrong_key": "not a list"}'])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_statement_count_mismatch():
    fake_llm = FakeStructuredChatModel(
        responses=['{"hypothesis_statements": [{"statements": ["statement1"]}]}']
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)


def test_prompt_contains_hypotheses():
    segments = ["hypothesis_segment1", "hypothesis_segment2"]
    fake_llm = FakeStructuredChatModel(
        responses=[
            '{"hypothesis_statements": [{"statements": ["s1"]}, {"statements": ["s2"]}]}'
        ]
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    extractor.extract(
        [SegmentedText(segments, [" "] * (len(segments) - 1))],
        show_progress_bar=False,
    )

    assert len(fake_llm.received_messages) == 1
    expected_prompt = get_statement_prompt("function_calling").format(
        **_make_statement_prompt_input(segments)
    )
    assert fake_llm.received_messages[0][-1].content == expected_prompt


def test_invoke_raises_exception():
    fake_llm = FakeStructuredChatModel(responses=[""])
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
