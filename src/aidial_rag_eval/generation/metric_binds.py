import dataclasses
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.inference import calculate_batch_inference
from aidial_rag_eval.generation.refusal import calculate_batch_refusal
from aidial_rag_eval.generation.types import InferenceMetricBind, RefusalMetricBind
from aidial_rag_eval.types import MergedColumns

C2A_INFERENCE_PREFIX = "ctx_ans_"
A2GT_INFERENCE_PREFIX = "ans_gt_"
GT2A_INFERENCE_PREFIX = "gt_ans_"

ANSWER_REFUSAL_PREFIX = "answer_"
GT_ANSWER_REFUSAL_PREFIX = "ground_truth_"


def _get_column_as_list(dataframe: pd.DataFrame, column: str) -> List[Any]:
    return [x.tolist() if isinstance(x, np.ndarray) else x for x in dataframe[column]]


def _wrapped_dataframe_inference(
    df_merged: pd.DataFrame,
    premise_column: str,
    hypothesis_column: str,
    llm: BaseChatModel,
    prefix: str,
    question_column: Optional[str] = None,
    document_column: Optional[str] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    inference_returns = calculate_batch_inference(
        premises=_get_column_as_list(df_merged, premise_column),
        hypotheses=_get_column_as_list(df_merged, hypothesis_column),
        llm=llm,
        questions=(
            _get_column_as_list(df_merged, question_column)
            if question_column is not None
            else None
        ),
        list_documents=(
            _get_column_as_list(df_merged, document_column)
            if document_column is not None
            else None
        ),
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
    )
    return pd.DataFrame(
        [dataclasses.asdict(inference_return) for inference_return in inference_returns]
    ).add_prefix(prefix)


def _wrapped_dataframe_refusal(
    df_merged: pd.DataFrame,
    answer_column: str,
    llm: BaseChatModel,
    prefix: str,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    refusal_returns = calculate_batch_refusal(
        answers=_get_column_as_list(df_merged, answer_column),
        llm=llm,
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
    )
    return pd.DataFrame(
        [dataclasses.asdict(refusal) for refusal in refusal_returns]
    ).add_prefix(prefix)


CONTEXT_TO_ANSWER_INFERENCE = InferenceMetricBind(
    premise_column=MergedColumns.CONTEXT,
    hypothesis_column=MergedColumns.ANSWER,
    prefix=C2A_INFERENCE_PREFIX,
    use_question=False,
    document_column=MergedColumns.DOCUMENTS,
)

ANSWER_TO_GROUND_TRUTH_INFERENCE = InferenceMetricBind(
    premise_column=MergedColumns.ANSWER,
    hypothesis_column=MergedColumns.GROUND_TRUTH_ANSWER,
    prefix=A2GT_INFERENCE_PREFIX,
    use_question=True,
    document_column=MergedColumns.DOCUMENTS,
)

GROUND_TRUTH_TO_ANSWER_INFERENCE = InferenceMetricBind(
    premise_column=MergedColumns.GROUND_TRUTH_ANSWER,
    hypothesis_column=MergedColumns.ANSWER,
    prefix=GT2A_INFERENCE_PREFIX,
    use_question=True,
    document_column=MergedColumns.DOCUMENTS,
)

ANSWER_REFUSAL = RefusalMetricBind(
    answer_column=MergedColumns.ANSWER,
    prefix=ANSWER_REFUSAL_PREFIX,
)

GROUND_TRUTH_REFUSAL = RefusalMetricBind(
    answer_column=MergedColumns.GROUND_TRUTH_ANSWER,
    prefix=GT_ANSWER_REFUSAL_PREFIX,
)
