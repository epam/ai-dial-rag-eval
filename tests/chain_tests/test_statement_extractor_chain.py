# flake8: noqa: E501
from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
)
from aidial_rag_eval.generation.models.statement_extractor.statement_extractor_template import (
    HypothesisStatements,
    StatementsOutput,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from tests.chain_tests.fake_models import FakeStructuredChatModel

EXPECTED_STATEMENT_PROMPT = (
    "\nBreak down each hypothesis into statements, if hypothesis is complex. Else return hypothesis as a single statement.\n"
    "\n"
    "A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.\n"
    "\n"
    "Single words, signs, numbers, links, etc. are not statements.\n"
    "\n"
    "Request:\n"
    "Hypotheses:\n"
    "\n"
    "<hypothesis1> hypothesis_segment1 </hypothesis1>\n"
    "\n"
    "<hypothesis2> hypothesis_segment2 </hypothesis2>\n"
    "\n"
    "\n"
    "IMPORTANT: Complete this entire task in a SINGLE response. Call the tool EXACTLY ONCE with ALL results in that one call."
)


def test_valid_json_response():
    fake_llm = FakeStructuredChatModel(
        responses=[
            (
                EXPECTED_STATEMENT_PROMPT,
                StatementsOutput(
                    hypothesis_statements=[
                        HypothesisStatements(statements=["statement11"]),
                        HypothesisStatements(statements=["statement21"]),
                    ]
                ),
            )
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


def test_statement_count_mismatch():
    fake_llm = FakeStructuredChatModel(
        responses=[
            StatementsOutput(
                hypothesis_statements=[HypothesisStatements(statements=["statement1"])]
            )
        ]
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
            StatementsOutput(
                hypothesis_statements=[
                    HypothesisStatements(statements=["s1"]),
                    HypothesisStatements(statements=["s2"]),
                ]
            )
        ]
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    extractor.extract(
        [SegmentedText(segments, [" "] * (len(segments) - 1))],
        show_progress_bar=False,
    )


def test_invoke_raises_exception():
    fake_llm = FakeStructuredChatModel(side_effect=Exception("LLM invoke failed"))
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract(
        [SegmentedText(hypothesis_segments, [" "] * (len(hypothesis_segments) - 1))],
        show_progress_bar=False,
    )[0]
    assert isinstance(result, ErrorInfo)
