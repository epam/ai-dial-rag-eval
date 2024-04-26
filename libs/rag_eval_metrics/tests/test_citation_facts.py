import numpy as np
import pandas as pd

from rag_eval_metrics.facts.citation import CitationMatcher
from rag_eval_metrics.facts.match_facts import match_facts


def test_match_facts():
    facts = ["fact1", "fact2", "fact3", "fact4"]
    context = ["fact1 and fact2", "some text, fact3, some more text", "some text"]
    facts_ranks, context_relevance = CitationMatcher.match_facts(facts, context)

    assert np.array_equal(facts_ranks, np.array([0, 0, 1, -1]))
    assert np.array_equal(context_relevance, np.array([2, 1, 0]))


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
        matched=CitationMatcher,
        axis=1,
        result_type="expand",
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "facts_ranks": [[-1], [0], [0, 0]],
                "context_relevance": [[0, 0], [1], [2, 0]],
            }
        ),
    )
