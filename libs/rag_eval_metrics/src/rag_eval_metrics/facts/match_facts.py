import pandas as pd

from rag_eval_metrics.types import Matcher


def match_facts(row: pd.Series, matched: Matcher) -> pd.Series:
    result_row = matched.match_facts(row.facts, row.context)
    return pd.Series(result_row._asdict())
