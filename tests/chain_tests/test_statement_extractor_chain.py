from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
)


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

    result = extractor.extract([hypothesis_segments], show_progress_bar=False)

    assert result == [[["statement11"], ["statement21"]]]


def test_invalid_json_response():
    fake_llm = FakeListChatModel(responses=["not valid json"])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract([hypothesis_segments], show_progress_bar=False)

    assert result == [
        [[hypothesis_segment] for hypothesis_segment in hypothesis_segments]
    ]


def test_json_wrong_structure():
    fake_llm = FakeListChatModel(responses=['{"wrong_key": "not a list"}'])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment2"]

    result = extractor.extract([hypothesis_segments], show_progress_bar=False)

    assert result == [
        [[hypothesis_segment] for hypothesis_segment in hypothesis_segments]
    ]


def test_statement_count_mismatch():
    fake_llm = FakeListChatModel(
        responses=['{"hypothesis_statements": [{"statements": ["statement1"]}]}']
    )
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract([hypothesis_segments], show_progress_bar=False)

    assert result == [
        [[hypothesis_segment] for hypothesis_segment in hypothesis_segments]
    ]


def test_empty_response():
    fake_llm = FakeListChatModel(responses=[""])
    extractor = LLMStatementExtractor(model=fake_llm, max_concurrency=1)

    hypothesis_segments = ["hypothesis_segment1", "hypothesis_segment1"]

    result = extractor.extract([hypothesis_segments], show_progress_bar=False)

    assert result == [
        [[hypothesis_segment] for hypothesis_segment in hypothesis_segments]
    ]
