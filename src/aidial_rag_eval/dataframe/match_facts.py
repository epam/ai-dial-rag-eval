import pandas as pd

from aidial_rag_eval.facts.citation import CitationMatcher
from aidial_rag_eval.types import Matcher

DEFAULT_MATCHER = CitationMatcher

KEY_COLUMNS = ["documents", "question"]
GROUND_TRUTH_COLUMNS = KEY_COLUMNS + ["facts"]
ANSWERS_COLUMNS = KEY_COLUMNS + ["context"]


def match_facts(row: pd.Series, matcher: Matcher) -> pd.Series:
    result_row = matcher.match_facts(row.facts, row.context)
    return pd.Series(result_row._asdict())


def match_facts_dataframe(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    ground_truth_copy = ground_truth.copy()
    ground_truth_copy["documents"] = ground_truth_copy["documents"].apply(
        lambda x: frozenset(x)
    )
    answers_copy = answers.copy()
    answers_copy["documents"] = answers_copy["documents"].apply(lambda x: frozenset(x))

    data = pd.merge(
        ground_truth_copy,
        answers_copy,
        on=KEY_COLUMNS,
    )
    data["documents"] = ground_truth.loc[data.index, "documents"]

    result = data.apply(
        match_facts,
        matcher=matcher,
        axis=1,
        result_type="expand",
    )

    return pd.merge(data, result, left_index=True, right_index=True)
