import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeRecordingChatModel

# flake8: noqa: E501
EXPECTED_DECONTEXTUALIZATION_PROMPT = """
The task is to replace all pronouns in segments with their corresponding nouns or proper names when their referents are known.
You will receive segments.
If a segment is nonsensical, a reference, link, or meaningless, return it unchanged.
If unsure what to do with segment, return the original segment.
Only perform the task; do not shorten, simplify, or correct errors.
Do not provide explanations.

For example:
[
  "My mom is a good person.",
  "She always takes care of me."
]
the expected output is:
{
  "segments": [
    "My mom is a good person.",
    "My mom always takes care of me."
  ]
}

Important: the response must have the same number of segments, split the same way.

List of input segments (JSON array of strings, one segment per element):
[
  "John went to the store.",
  "He bought milk."
]"""


def test_valid_json_response():
    fake_llm = FakeListChatModel(
        responses=[
            json.dumps({"segments": ["John went to the store.", "John bought milk."]})
        ]
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
    fake_llm = FakeListChatModel(
        responses=[json.dumps({"segments": ["only one segment"]})]
    )
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
    fake_llm = FakeRecordingChatModel(
        responses=[], side_effect=Exception("LLM invoke failed")
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_single_segment_skips_llm():
    fake_llm = FakeRecordingChatModel(responses=[])
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    result = converter.transform_texts(
        [SegmentedText(segments=["Only one segment."], delimiters=[])],
        show_progress_bar=False,
    )[0]

    assert not isinstance(result, ErrorInfo)
    assert result.segments == ["Only one segment."]
    assert fake_llm.recorded_inputs == []


def test_prompt_contains_segments():
    segments = ["John went to the store.", "He bought milk."]
    fake_llm = FakeRecordingChatModel(
        responses=[
            json.dumps({"segments": ["John went to the store.", "John bought milk."]})
        ]
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    converter.transform_texts(
        [SegmentedText(segments=segments, delimiters=[" "])],
        show_progress_bar=False,
    )

    assert fake_llm.recorded_inputs[0][0].content == EXPECTED_DECONTEXTUALIZATION_PROMPT
