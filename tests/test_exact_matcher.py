import json

import numpy as np

from aidial_rag_eval.facts.exact import ExactStringMatcher


def test_exact_match_facts():
    facts = ["page2", "page3"]
    context = ["page3", "page21", "_page2"]
    facts_ranks, context_relevance, context_highlight = ExactStringMatcher.match_facts(
        facts, context
    )

    assert np.array_equal(facts_ranks, np.array([-1, 0]))
    assert np.array_equal(context_relevance, np.array([1, 0, 0]))

    assert np.array_equal(
        context_highlight,
        [
            json.dumps(
                {
                    "match": [
                        {"text": "page3", "facts": [1]},
                    ]
                }
            ),
            json.dumps(
                {
                    "match": [
                        {"text": "page21", "facts": []},
                    ]
                }
            ),
            json.dumps(
                {
                    "match": [
                        {"text": "_page2", "facts": []},
                    ]
                }
            ),
        ],
    )
