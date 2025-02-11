from typing import Callable, Dict, List, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.inference import calculate_batch_inference
from aidial_rag_eval.generation.refusal import calculate_batch_refusal
from aidial_rag_eval.generation.types import MetricBind
from aidial_rag_eval.types import MergedColumns

C2A_INFERENCE_PREFIX = "ctx_ans_"
A2GT_INFERENCE_PREFIX = "ans_gt_"
GT2A_INFERENCE_PREFIX = "gt_ans_"

ANSWER_REFUSAL_PREFIX = "answer_"
GT_ANSWER_REFUSAL_PREFIX = "ground_truth_"


def _get_column_as_list_str(dataframe: pd.DataFrame, column: str) -> List[str]:
    list_str = dataframe[column].to_list()
    assert isinstance(list_str, list)
    return list_str


def _get_column_as_list_list_str(
    dataframe: pd.DataFrame, column: str
) -> List[List[str]]:
    list_str = dataframe[column].to_list()
    assert isinstance(list_str, list)
    return list_str


def _wrapped_dataframe_inference(
    data: pd.DataFrame,
    premise_column: str,
    hypothesis_column: str,
    llm: BaseChatModel,
    prefix: str,
    question_column: Optional[str] = None,
    document_column: Optional[str] = None,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    inference_returns = calculate_batch_inference(
        _get_column_as_list_str(data, premise_column),
        _get_column_as_list_str(data, hypothesis_column),
        llm,
        (
            _get_column_as_list_str(data, question_column)
            if question_column is not None
            else None
        ),
        (
            _get_column_as_list_list_str(data, document_column)
            if document_column is not None
            else None
        ),
        max_concurrency=max_concurrency,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return pd.DataFrame(
        [vars(inference_return) for inference_return in inference_returns]
    ).add_prefix(prefix)


def _wrapped_dataframe_refusal(
    data: pd.DataFrame,
    answer_column: str,
    llm: BaseChatModel,
    prefix: str,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    refusal_returns = calculate_batch_refusal(
        _get_column_as_list_str(data, answer_column),
        llm,
        max_concurrency=max_concurrency,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return pd.DataFrame([vars(refusal) for refusal in refusal_returns]).add_prefix(
        prefix
    )


def context_to_answer_inference(
    data, llm, max_concurrency, batch_size, show_progress_bar, **kwargs
) -> pd.DataFrame:
    return _wrapped_dataframe_inference(
        data,
        MergedColumns.CONTEXT,
        MergedColumns.ANSWER,
        llm,
        C2A_INFERENCE_PREFIX,
        None,
        MergedColumns.DOCUMENTS,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )


def answer_to_ground_truth_inference(
    data, llm, max_concurrency, batch_size, show_progress_bar, **kwargs
) -> pd.DataFrame:
    return _wrapped_dataframe_inference(
        data,
        MergedColumns.ANSWER,
        MergedColumns.GROUND_TRUTH_ANSWER,
        llm,
        A2GT_INFERENCE_PREFIX,
        MergedColumns.QUESTION,
        MergedColumns.DOCUMENTS,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )


def ground_truth_to_answer_inference(
    data, llm, max_concurrency, batch_size, show_progress_bar, **kwargs
) -> pd.DataFrame:
    return _wrapped_dataframe_inference(
        data,
        MergedColumns.GROUND_TRUTH_ANSWER,
        MergedColumns.ANSWER,
        llm,
        GT2A_INFERENCE_PREFIX,
        MergedColumns.QUESTION,
        MergedColumns.DOCUMENTS,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )


def answer_refusal(
    data, llm, max_concurrency, batch_size, show_progress_bar, **kwargs
) -> pd.DataFrame:
    return _wrapped_dataframe_refusal(
        data,
        MergedColumns.ANSWER,
        llm,
        ANSWER_REFUSAL_PREFIX,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )


def ground_truth_refusal(
    data, llm, max_concurrency, batch_size, show_progress_bar, **kwargs
) -> pd.DataFrame:
    return _wrapped_dataframe_refusal(
        data,
        MergedColumns.GROUND_TRUTH_ANSWER,
        llm,
        GT_ANSWER_REFUSAL_PREFIX,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )


metric_binds_dict: Dict[MetricBind, Callable] = {
    "context_to_answer_inference": context_to_answer_inference,
    "answer_to_ground_truth_inference": answer_to_ground_truth_inference,
    "ground_truth_to_answer_inference": ground_truth_to_answer_inference,
    "answer_refusal": answer_refusal,
    "ground_truth_refusal": ground_truth_refusal,
}
metric_bind_keys: List[MetricBind] = list(metric_binds_dict.keys())
