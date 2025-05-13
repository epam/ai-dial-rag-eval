from typing import List, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.dataframe.match_facts import (
    DEFAULT_MATCHER,
    apply_matcher_to_merged_dataframe,
)
from aidial_rag_eval.dataframe.merge import merge_ground_truth_and_answers
from aidial_rag_eval.generation.metric_binds import metric_binds_dict
from aidial_rag_eval.generation.types import MetricBind, inference_column
from aidial_rag_eval.retrieval.metrics import (
    calculate_metrics as calculate_metrics_by_row,
)
from aidial_rag_eval.retrieval.types import Matcher
from aidial_rag_eval.types import MergedColumns


def apply_metrics_to_matched_results(match_result_data: pd.DataFrame) -> pd.DataFrame:
    retrieval_metrics = match_result_data.apply(
        calculate_metrics_by_row,
        axis=1,
        result_type="expand",
    )
    assert isinstance(retrieval_metrics, pd.DataFrame)
    return retrieval_metrics


def calculate_metrics(match_result_data: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        match_result_data,
        apply_metrics_to_matched_results(match_result_data),
        left_index=True,
        right_index=True,
    )


def calculate_retrieval_metrics(
    df_merged: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    """
    Calculates RAG evaluation retrieval metrics from df_merged dataframe.

    Parameters
    -----------
    df_merged : pd.DataFrame
        merged ground truth answers and collected answers dataframes
        The structure of the df_merged
        is described in `aidial_rag_eval.types.MergedColumns`.

    matcher : Matcher, default=DEFAULT_MATCHER
        An object responsible for matching facts
        from the ground truth to the context in the answers.

    Returns
    ------------
    pd.DataFrame
        Returns retrieval metrics dataframe.
    """
    matched_result = apply_matcher_to_merged_dataframe(df_merged, matcher)
    matched_result_with_df = pd.merge(
        df_merged, matched_result, left_index=True, right_index=True
    )
    retrieval_metrics = apply_metrics_to_matched_results(matched_result_with_df)
    return pd.merge(
        matched_result, retrieval_metrics, left_index=True, right_index=True
    )


def create_retrieval_metrics_report(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
) -> pd.DataFrame:
    """
    Calculates RAG evaluation retrieval metrics from input dataframes.

    Parameters
    -----------
    ground_truth : pd.DataFrame
        contains the ground truth answers
        The structure of the ground truth
        is described in `aidial_rag_eval.types.GroundTruthAnswers`.

    answers : pd.DataFrame
        contains the collected answers
        The structure of the collected answers
        is described in `aidial_rag_eval.types.CollectedAnswers`.

    matcher : Matcher, default=DEFAULT_MATCHER
        An object responsible for matching facts
        from the ground truth to the context in the answers.

    Returns
    ------------
    pd.DataFrame
        Returns merged ground_truth dataframe,
        answers dataframe with retrieval metrics dataframe.
    """
    df_merged = merge_ground_truth_and_answers(ground_truth, answers)
    return pd.merge(
        df_merged,
        calculate_retrieval_metrics(df_merged, matcher),
        left_index=True,
        right_index=True,
    )


def calculate_generation_metrics(
    df_merged: pd.DataFrame,
    llm: BaseChatModel,
    metric_binds: List[MetricBind],
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Calculates RAG evaluation generation metrics from df_merged dataframe.

    Parameters
    -----------
    df_merged : pd.DataFrame
        merged ground truth answers and collected answers dataframes
        The structure of the df_merged
        is described in `aidial_rag_eval.types.MergedColumns`.

    llm : BaseChatModel, optional, default=None
        A Langchain chat model used for calculating generation metrics.

    metric_binds : List[MetricBind], optional, default=None
        A list of string constants from `aidial_rag_eval.metric_binds`
        specifying the generation metrics.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    batch_size : int, default=6
        The maximum number objects are processed in a single prompt for simple tasks.

    show_progress_bar : bool, default=True
        To display a progress bar during LLM requests.

    Returns
    ------------
    pd.DataFrame
        Returns generation metrics dataframe.
    """
    df_merged_copy = df_merged.copy()
    df_merged_copy[MergedColumns.JOINED_CONTEXT] = df_merged_copy[
        MergedColumns.CONTEXT
    ].apply(lambda x: "\n".join(x))
    metric_results = dict()
    for metric_bind in metric_binds:
        metric_results.update(
            metric_binds_dict[metric_bind](
                df_merged=df_merged_copy,
                llm=llm,
                max_concurrency=max_concurrency,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            ).to_dict(orient="series")
        )
    df_metrics = pd.DataFrame(data=metric_results)
    nli_columns = [
        column for column in metric_results.keys() if column.endswith(inference_column)
    ]
    if nli_columns:
        sub_df_nli = df_metrics[nli_columns]
        df_metrics["mean_" + inference_column] = sub_df_nli.mean(1)
        df_metrics["median_" + inference_column] = sub_df_nli.median(
            1
        )  # pyright: ignore # noqa

    return df_metrics


def create_generation_metrics_report(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    llm: BaseChatModel,
    metric_binds: List[MetricBind],
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Calculates RAG evaluation generation metrics from input dataframes.

    Parameters
    -----------
    ground_truth : pd.DataFrame
        contains the ground truth answers
        The structure of the ground truth
        is described in `aidial_rag_eval.types.GroundTruthAnswers`.

    answers : pd.DataFrame
        contains the collected answers
        The structure of the collected answers
        is described in `aidial_rag_eval.types.CollectedAnswers`.

    llm : BaseChatModel, optional, default=None
        A Langchain chat model used for calculating generation metrics.

    metric_binds : List[MetricBind], optional, default=None
        A list of string constants from `aidial_rag_eval.metric_binds`
        specifying the generation metrics.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    batch_size : int, default=6
        The maximum number objects are processed in a single prompt for simple tasks.

    show_progress_bar : bool, default=True
        To display a progress bar during LLM requests.

    Returns
    ------------
    pd.DataFrame
        Returns merged ground_truth dataframe,
        answers dataframe with generation metrics dataframe.
    """
    df_merged = merge_ground_truth_and_answers(ground_truth, answers)
    df_metrics = calculate_generation_metrics(
        df_merged,
        llm,
        metric_binds,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )
    return pd.merge(df_merged, df_metrics, left_index=True, right_index=True)


def create_rag_eval_metrics_report(
    ground_truth: pd.DataFrame,
    answers: pd.DataFrame,
    matcher: Matcher = DEFAULT_MATCHER,
    llm: Optional[BaseChatModel] = None,
    metric_binds: Optional[List[MetricBind]] = None,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Calculates RAG evaluation metrics from input dataframes.

    Parameters
    -----------
    ground_truth : pd.DataFrame
        contains the ground truth answers
        The structure of the ground truth
        is described in `aidial_rag_eval.types.GroundTruthAnswers`.

    answers : pd.DataFrame
        contains the collected answers
        The structure of the collected answers
        is described in `aidial_rag_eval.types.CollectedAnswers`.

    matcher : Matcher, default=DEFAULT_MATCHER
        An object responsible for matching facts
        from the ground truth to the context in the answers.

    llm : BaseChatModel, optional, default=None
        A Langchain chat model used for calculating generation metrics.

    metric_binds : List[MetricBind], optional, default=None
        A list of string constants from `aidial_rag_eval.metric_binds`
        specifying the generation metrics.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    batch_size : int, default=6
        The maximum number objects are processed in a single prompt for simple tasks.

    show_progress_bar : bool, default=True
        To display a progress bar during LLM requests.

    Returns
    ------------
    pd.DataFrame
        Returns merged ground_truth dataframe, answers dataframe with metrics dataframe.
    """
    df_merged = merge_ground_truth_and_answers(ground_truth, answers)
    retrieval_metrics = calculate_retrieval_metrics(df_merged, matcher)
    if not metric_binds:
        return pd.merge(df_merged, retrieval_metrics, left_index=True, right_index=True)
    if llm is None:
        raise ValueError("Argument 'llm' is required.")
    generation_metrics = calculate_generation_metrics(
        df_merged,
        llm,
        metric_binds,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )
    return pd.concat([df_merged, retrieval_metrics, generation_metrics], axis=1)
