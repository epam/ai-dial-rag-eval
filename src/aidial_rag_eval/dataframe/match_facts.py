import pandas as pd

from aidial_rag_eval.dataframe.merge import merge_ground_truth_and_answers
from aidial_rag_eval.facts.citation import CitationMatcher
from aidial_rag_eval.retrieval.types import Matcher

DEFAULT_MATCHER = CitationMatcher


def match_facts(row: pd.Series, matcher: Matcher) -> pd.Series:
    result_row = matcher.match_facts(row.facts, row.context)
    return pd.Series(result_row._asdict())


def apply_matcher_to_merged_dataframe(
    df_merged: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    matched_results = df_merged.apply(
        match_facts,
        matcher=matcher,
        axis=1,
        result_type="expand",
    )
    assert isinstance(matched_results, pd.DataFrame)
    return matched_results


def match_facts_dataframe(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    df_merged = merge_ground_truth_and_answers(ground_truth, answers)
    return pd.merge(
        df_merged,
        apply_matcher_to_merged_dataframe(df_merged, matcher),
        left_index=True,
        right_index=True,
    )
