from typing import Any, List, Optional, Union

import pandas as pd
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.dataframe import create_rag_eval_metrics_report
from aidial_rag_eval.dataframe.match_facts import DEFAULT_MATCHER
from aidial_rag_eval.dataset import Dataset, source_dataset
from aidial_rag_eval.generation.types import MetricBind
from aidial_rag_eval.retrieval.types import Matcher
from aidial_rag_eval.utils import get_tools_versions


def evaluate(
    ground_truth: Union[str, Dataset],
    answers: Union[str, Dataset],
    dest: str,
    matcher: Matcher = DEFAULT_MATCHER,
    llm: Optional[BaseChatModel] = None,
    fs: Any = None,
    metric_binds: Optional[List[MetricBind]] = None,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> Dataset:
    """
    Calculates RAG evaluation metrics from input
    Parquet datasets and writes the results to the specified destination.

    Parameters
    -----------
    ground_truth : Union[str, Dataset]
        Path to the Parquet file containing ground truth answers or
        a Dataset object specifying the Parquet file with ground truth answers.
        The structure of the ground truth Parquet file
        is described in `aidial_rag_eval.types.GroundTruthAnswers`.

    answers : Union[str, Dataset]
        Path to the Parquet file containing collected answers or
        a Dataset object specifying the Parquet file with collected answers.
        The structure of the collected answers Parquet file
        is described in `aidial_rag_eval.types.CollectedAnswers`.

    dest : str
        Path for the output file in Parquet format.

    matcher : Matcher, default=DEFAULT_MATCHER
        An object responsible for matching facts
        from the ground truth to the context in the answers.

    llm : BaseChatModel, optional, default=None
        A Langchain chat model used for calculating generation metrics.

    fs : Any, optional, default=None
        File system to be used.

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
    Dataset
        Returns global statistics for each metric.
    """
    ground_truth_dataset = source_dataset(ground_truth)
    answers_dataset = source_dataset(answers)

    ground_truth_df = ground_truth_dataset.read_dataframe(filesystem=fs)
    answers_df = answers_dataset.read_dataframe(filesystem=fs)

    df_final = create_rag_eval_metrics_report(
        ground_truth_df,
        answers_df,
        matcher,
        llm,
        metric_binds=metric_binds,
        max_concurrency=max_concurrency,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    aggregated_metrics = df_final.mean(numeric_only=True)
    assert isinstance(aggregated_metrics, pd.Series)

    metrics = Dataset.write_dataframe(
        df_final,
        dest,
        sources=[ground_truth_dataset, answers_dataset],
        tools=get_tools_versions(),
        metrics=aggregated_metrics.to_dict(),
        statistics={
            "Ground truth size": len(ground_truth_df),
            "Answers size": len(answers_df),
            "Evaluation data size": len(df_final),
        },
        filesystem=fs,
    )
    return metrics
