import json
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.converters.decontextualization_template import (
    get_decontextualization_prompt,
)
from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeStructuredChatModel


def test_valid_json_response():
    fake_llm = FakeStructuredChatModel(
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
    fake_llm = FakeStructuredChatModel(responses=["not a valid json at all"])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_json_missing_segments_key():
    fake_llm = FakeStructuredChatModel(
        responses=['{"wrong_key": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_segment_count_mismatch():
    fake_llm = FakeStructuredChatModel(responses=['{"segments": ["only one segment"]}'])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_prompt_contains_segments():
    segments = ["John went to the store.", "He bought milk."]
    fake_llm = FakeStructuredChatModel(
        responses=['{"segments": ["John went to the store.", "John bought milk."]}']
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    converter.transform_texts(
        [SegmentedText(segments=segments, delimiters=[" "])],
        show_progress_bar=False,
    )

    assert len(fake_llm.received_messages) == 1
    expected_prompt = get_decontextualization_prompt("function_calling").format(
        sentences_str=json.dumps(segments)
    )
    assert fake_llm.received_messages[0][-1].content == expected_prompt


def test_single_segment_skips_llm():
    fake_llm = FakeStructuredChatModel(responses=[])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    result = converter.transform_texts(
        [SegmentedText(segments=["Only one segment."], delimiters=[])],
        show_progress_bar=False,
    )[0]

    assert not isinstance(result, ErrorInfo)
    assert result.segments == ["Only one segment."]


def test_invoke_raises_exception():
    fake_llm = FakeStructuredChatModel(responses=[""])

    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    with patch.object(
        FakeListChatModel, "batch", side_effect=Exception("LLM invoke failed")
    ):
        result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
        assert isinstance(result, ErrorInfo)
