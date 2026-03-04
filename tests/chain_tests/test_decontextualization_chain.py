from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


def test_valid_json_response():
    fake_llm = FakeListChatModel(
        responses=['{"segments": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    converter.transform_texts([segmented_text], show_progress_bar=False)

    assert segmented_text.segments == ["John went to the store.", "John bought milk."]


def test_invalid_json_response():
    fake_llm = FakeListChatModel(responses=["not a valid json at all"])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )
    original_segments = segmented_text.segments.copy()

    converter.transform_texts([segmented_text], show_progress_bar=False)

    assert segmented_text.segments == original_segments


def test_json_missing_segments_key():
    fake_llm = FakeListChatModel(
        responses=['{"wrong_key": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )
    original_segments = segmented_text.segments.copy()

    converter.transform_texts([segmented_text], show_progress_bar=False)

    assert segmented_text.segments == original_segments


def test_segment_count_mismatch():
    fake_llm = FakeListChatModel(responses=['{"segments": ["only one segment"]}'])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )
    original_segments = segmented_text.segments.copy()

    converter.transform_texts([segmented_text], show_progress_bar=False)

    assert segmented_text.segments == original_segments


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )
    original_segments = segmented_text.segments.copy()

    converter.transform_texts([segmented_text], show_progress_bar=False)

    assert segmented_text.segments == original_segments


def test_invoke_raises_exception():
    fake_llm = FakeListChatModel(responses=[""])

    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    with patch.object(
        FakeListChatModel, "invoke", side_effect=Exception("LLM invoke failed")
    ):
        try:
            converter.transform_texts([segmented_text], show_progress_bar=False)
            raise AssertionError("Expected exception was not raised")
        except Exception as e:
            assert str(e) == "LLM invoke failed"
