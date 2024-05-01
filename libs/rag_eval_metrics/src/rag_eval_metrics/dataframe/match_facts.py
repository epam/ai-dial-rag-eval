import pandas as pd

from rag_eval_metrics.facts.citation import CitationMatcher
from rag_eval_metrics.types import Matcher

DEFAULT_MATCHER = CitationMatcher


def match_facts(row: pd.Series, matcher: Matcher) -> pd.Series:
    result_row = matcher.match_facts(row.facts, row.context)
    return pd.Series(result_row._asdict())


def match_facts_dataframe(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    data = pd.merge(
        ground_truth,
        answers,
        on=["question"],
    )

    result = data.apply(
        match_facts,
        matcher=matcher,
        axis=1,
        result_type="expand",
    )

    result = pd.merge(data, result, left_index=True, right_index=True)

    return result
