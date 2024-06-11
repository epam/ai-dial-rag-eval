import pandas as pd

from aidial_rag_eval.metrics import calculate_metrics as calculate_metrics_by_row


def calculate_metrics(match_result_data: pd.DataFrame) -> pd.DataFrame:
    result = match_result_data.apply(
        calculate_metrics_by_row,
        axis=1,
        result_type="expand",
    )

    return pd.merge(match_result_data, result, left_index=True, right_index=True)
