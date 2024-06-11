import json

import numpy as np
import pandas as pd
import pytest

from aidial_rag_eval.dataframe.match_facts import match_facts
from aidial_rag_eval.facts.citation import CitationMatcher


def test_match_facts():
    facts = ["fact1", "fact2", "fact3", "fact4"]
    context = ["fact1 and fact2", "some text, fact3, some more text", "some text"]
    facts_ranks, context_relevance, context_highlight = CitationMatcher.match_facts(
        facts, context
    )

    assert np.array_equal(facts_ranks, np.array([0, 0, 1, -1]))
    assert np.array_equal(context_relevance, np.array([2, 1, 0]))

    assert np.array_equal(
        context_highlight,
        [
            json.dumps(
                {
                    "match": [
                        {"text": "fact1", "facts": [0]},
                        {"text": " and ", "facts": []},
                        {"text": "fact2", "facts": [1]},
                    ]
                }
            ),
            json.dumps(
                {
                    "match": [
                        {"text": "some text, ", "facts": []},
                        {"text": "fact3", "facts": [2]},
                        {"text": ", some more text", "facts": []},
                    ]
                }
            ),
            json.dumps(
                {
                    "match": [
                        {"text": "some text", "facts": []},
                    ]
                }
            ),
        ],
    )


def test_dataframe():
    data = pd.DataFrame(
        [
            {"facts": ["fact1"], "context": ["some text", "some text"]},
            {"facts": ["fact2"], "context": ["some text with fact2"]},
            {
                "facts": ["fact3", "fact4"],
                "context": ["some text with fact3 and fact4", "some extra text"],
            },
        ]
    )

    result = data.apply(
        match_facts,
        matcher=CitationMatcher,
        axis=1,
        result_type="expand",
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "facts_ranks": [[-1], [0], [0, 0]],
                "context_relevance": [[0, 0], [1], [2, 0]],
                "context_highlight": [
                    [
                        json.dumps(
                            {
                                "match": [
                                    {"text": "some text", "facts": []},
                                ]
                            }
                        ),
                        json.dumps(
                            {
                                "match": [
                                    {"text": "some text", "facts": []},
                                ]
                            }
                        ),
                    ],
                    [
                        json.dumps(
                            {
                                "match": [
                                    {"text": "some text with ", "facts": []},
                                    {"text": "fact2", "facts": [0]},
                                ]
                            }
                        )
                    ],
                    [
                        json.dumps(
                            {
                                "match": [
                                    {"text": "some text with ", "facts": []},
                                    {"text": "fact3", "facts": [0]},
                                    {"text": " and ", "facts": []},
                                    {"text": "fact4", "facts": [1]},
                                ]
                            }
                        ),
                        json.dumps(
                            {
                                "match": [
                                    {"text": "some extra text", "facts": []},
                                ]
                            }
                        ),
                    ],
                ],
            }
        ),
    )


def test_context_highlight():
    _, _, highlight = CitationMatcher.match_facts(
        ["bcdefghij", "cdefghi", "cd", "hi", "l"],
        ["abcdefghijklm"],
    )
    # abcdefghijklm
    # -------------
    #  bcdefghij l
    #   cdefghi
    #   cd   hi

    assert len(highlight) == 1
    assert highlight[0] == json.dumps(
        {
            "match": [
                {"text": "a", "facts": []},
                {"text": "b", "facts": [0]},
                {"text": "cd", "facts": [0, 1, 2]},
                {"text": "efg", "facts": [0, 1]},
                {"text": "hi", "facts": [0, 1, 3]},
                {"text": "j", "facts": [0]},
                {"text": "k", "facts": []},
                {"text": "l", "facts": [4]},
                {"text": "m", "facts": []},
            ]
        }
    )


def test_context_highlight_chinese():
    _, _, highlight = CitationMatcher.match_facts(
        ["你好", "好", "你", "再见"],
        ["你好你再见"],
    )

    assert len(highlight) == 1
    assert highlight[0] == json.dumps(
        {
            "match": [
                {"text": "你", "facts": [0, 2]},
                {"text": "好", "facts": [0, 1]},
                {"text": "你", "facts": []},
                {"text": "再见", "facts": [3]},
            ]
        }
    )


def test_context_highlight_extra_spaces():
    _, _, highlight = CitationMatcher.match_facts(
        ["a", "b"],
        ["   a   b  "],
    )

    assert len(highlight) == 1
    assert highlight[0] == json.dumps(
        {
            "match": [
                {"text": "   ", "facts": []},
                {"text": "a", "facts": [0]},
                {"text": "   ", "facts": []},
                {"text": "b", "facts": [1]},
                {"text": "  ", "facts": []},
            ]
        }
    )


@pytest.mark.skip(reason="Need too check what is the expected behavior here")
def test_context_highlight_extra_spaces2():
    _, _, highlight = CitationMatcher.match_facts(
        ["   a       ", " b "],
        [" a   b  "],
    )

    assert len(highlight) == 1
    assert highlight[0] == json.dumps(
        {
            "match": [
                {"text": " a ", "facts": [0]},
                {"text": " ", "facts": []},
                {"text": " b ", "facts": [1]},
                {"text": " ", "facts": []},
            ]
        }
    )


def test_context_highlight_nonalhpanum():
    _, _, highlight = CitationMatcher.match_facts(
        ["@", "#", "()"],
        ["@#$%^&*()"],
    )

    assert len(highlight) == 1
    assert highlight[0] == json.dumps(
        {
            "match": [
                {"text": "@#$%^&*()", "facts": [0, 1, 2]},
            ]
        }
    )
