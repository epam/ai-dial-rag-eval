from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


def test_valid_json_response():
    fake_llm = FakeListChatModel(
        responses=['{"segments": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert not isinstance(result, ErrorInfo)
    assert result.segments == [
        "John went to the store.",
        "John bought milk.",
    ]


def test_invalid_json_response():
    fake_llm = FakeListChatModel(responses=["not a valid json at all"])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_json_missing_segments_key():
    fake_llm = FakeListChatModel(
        responses=['{"wrong_key": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_segment_count_mismatch():
    fake_llm = FakeListChatModel(responses=['{"segments": ["only one segment"]}'])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_invoke_raises_exception():
    fake_llm = FakeListChatModel(responses=[""])

    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    with patch.object(
        FakeListChatModel, "batch", side_effect=Exception("LLM invoke failed")
    ):
        result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
        assert isinstance(result, ErrorInfo)
