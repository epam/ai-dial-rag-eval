import pandas as pd

from rag_eval_metrics.metrics import calculate_metrics as calculate_metrics_by_row


def calculate_metrics(match_result_data: pd.DataFrame) -> pd.DataFrame:
    result = match_result_data.apply(
        calculate_metrics_by_row,
        axis=1,
        result_type="expand",
    )

    # pyright complains that result is pd.DataFrame | pd.Series
    assert isinstance(result, pd.DataFrame)
    return result
