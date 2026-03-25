from aidial_rag_eval.generation.models.converters.decontextualization_template import (
    DecontextualizationOutput,
)
from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeStructuredChatModel


EXPECTED_DECONTEXTUALIZATION_PROMPT = (
    "\nThe task is to replace all pronouns in segments with their corresponding nouns or proper names when their referents are known.\n"
    "You will receive segments.\n"
    "If a segment is nonsensical, a reference, link, or meaningless, return it unchanged.\n"
    "If unsure what to do with segment, return the original segment.\n"
    "Only perform the task; do not shorten, simplify, or correct errors.\n"
    "Do not provide explanations.\n"
    "\n"
    'For example: "My mom is a good person.", "She always takes care of me."\n'
    'should return segments: ["My mom is a good person.", "My mom always takes care of me."]\n'
    "\n"
    "Important: the response must have the same number of segments, split the same way.\n"
    "\n"
    "List of input segments:\n"
    '["John went to the store.", "He bought milk."]\n'
    "\n"
    "IMPORTANT: Complete this entire task in a SINGLE response. Call the tool EXACTLY ONCE with ALL results in that one call."
)


def test_valid_json_response():
    fake_llm = FakeStructuredChatModel(
        responses=[
            (
                EXPECTED_DECONTEXTUALIZATION_PROMPT,
                DecontextualizationOutput(
                    segments=["John went to the store.", "John bought milk."]
                ),
            )
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


def test_segment_count_mismatch():
    fake_llm = FakeStructuredChatModel(
        responses=[DecontextualizationOutput(segments=["only one segment"])]
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)


def test_prompt_contains_segments():
    segments = ["John went to the store.", "He bought milk."]
    fake_llm = FakeStructuredChatModel(
        responses=[
            DecontextualizationOutput(
                segments=["John went to the store.", "John bought milk."]
            )
        ]
    )
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    converter.transform_texts(
        [SegmentedText(segments=segments, delimiters=[" "])],
        show_progress_bar=False,
    )


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
    fake_llm = FakeStructuredChatModel(side_effect=Exception("LLM invoke failed"))
    converter = LLMNoPronounsConverter(model=fake_llm, max_concurrency=1)

    segmented_text = SegmentedText(
        segments=["John went to the store.", "He bought milk."], delimiters=[" "]
    )

    result = converter.transform_texts([segmented_text], show_progress_bar=False)[0]
    assert isinstance(result, ErrorInfo)
