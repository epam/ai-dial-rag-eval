import itertools
import json
from enum import Enum
from itertools import chain
from typing import List, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.llm_models.converter import LLMNoPronounsBatchConverter
from aidial_rag_eval.generation.llm_models.inference_scorer import LLMInferenceScorer
from aidial_rag_eval.generation.types import (
    Hypothesis,
    InferenceBatchItem,
    InferenceReturn,
    JoinedDocumentsName,
    Premise,
)
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from aidial_rag_eval.types import Documents, Question


class _InternalColumns(str, Enum):
    hypothesis_id = "q_num"
    hypothesis_split = "hypothesis_split"
    hypothesis = "hypothesis"
    premise = "premise"
    json = "json"
    nli = "inference"
    explanation = "explanation"


def _row_to_json(row: dict) -> str:
    return json.dumps(
        {k: (v if k != _InternalColumns.premise else [v]) for k, v in row.items()},
    )


def _json_to_highlight(row: dict) -> str:
    inference_json = row[_InternalColumns.json]
    segmented_text = row[_InternalColumns.hypothesis_split]
    highlight = dict()
    delimiter_highlight = dict()
    highlight["corpus"] = []
    delimiter_highlight["corpus"] = []
    delimiters = segmented_text.delimiters
    json_answer_list = json.loads(inference_json)
    for ans, delimiter in zip(json_answer_list, delimiters + [""]):
        highlight["corpus"].append(
            {
                "text": ans[_InternalColumns.hypothesis],
                "score": ans[_InternalColumns.nli] - 1,
                "title": ans[_InternalColumns.nli],
            }
        )
        delimiter_highlight["corpus"].append({"text": delimiter, "score": 0.0})
    highlight["corpus"] = list(
        itertools.chain.from_iterable(
            zip(highlight["corpus"], delimiter_highlight["corpus"])
        )
    )
    return json.dumps(highlight)


def _join_documents(documents: Documents) -> JoinedDocumentsName:
    return " ; ".join(documents)


def _make_input_batch(
    premises: List[Premise],
    segmented_hypothesis: List[SegmentedText],
    document_names: List[JoinedDocumentsName],
) -> List[InferenceBatchItem]:
    input_batch = list(
        chain.from_iterable(
            [
                [
                    InferenceBatchItem(
                        hypothesis_id=i,
                        premise=premises[i],
                        hypothesis_segment=hypothesis,
                        document_name=document_names[i],
                    )
                    for hypothesis in segmented_hypothesis[i].segments
                ]
                for i in range(len(segmented_hypothesis))
            ]
        )
    )
    return input_batch


def calculate_batch_inference(
    premises: List[Premise],
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> List[InferenceReturn]:
    """
    Calculates pairwise the inference of a hypotheses from a premises.

    Parameters
    -----------

        premises : List[str]
            The text of the premise from which the hypothesis will be inferred.

        hypotheses : List[str]
            The text of the hypothesis.

        llm : BaseChatModel
            The Langchain chat model used for calculating inference.

        questions : List[str], optional, default=None
            A questions related to the inference process as a part of the premise.

        list_documents : List[List[str]], optional, default=None
            A list of document names that used
            in the inference process as a part of the premises.

        max_concurrency : int, default=8
            The maximum number of concurrent requests to the LLM.

        batch_size : int, default=6
            The maximum number of objects processed in a single prompt for simple tasks.

        show_progress_bar : bool, default=True
            Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[InferenceReturn]
        Returns the list of inference,
        along with a JSON strings that explains how the inference was derived and
        highlights strings used for highlighting each segment of the each hypothesis.
    """

    converter = LLMNoPronounsBatchConverter(
        model=llm, batch_size=batch_size, max_concurrency=max_concurrency
    )
    scorer = LLMInferenceScorer(model=llm, max_concurrency=max_concurrency)

    segmented_hypotheses = [
        SegmentedText.from_text(text=hypothesis) for hypothesis in hypotheses
    ]
    if show_progress_bar:
        print("Converting hypothesis...")
    converter.transform_texts(segmented_hypotheses, show_progress_bar)
    if list_documents is None:
        document_names: List[JoinedDocumentsName] = [
            "Document name is not specified."
        ] * len(hypotheses)
    else:
        document_names = [_join_documents(docs) for docs in list_documents]
    if questions is not None:
        segmented_questions = [
            SegmentedText.from_text(text=question) for question in questions
        ]
        premises = [
            question_split.segments[-1] + "\n" + premise
            for question_split, premise in zip(segmented_questions, premises)
        ]
    input_batch = _make_input_batch(
        premises,
        segmented_hypotheses,
        document_names,
    )
    if show_progress_bar:
        print("Getting inference...")
    inference_scores = scorer.get_inference(
        input_batch,
        show_progress_bar,
    )

    df_pre_aggregation_scores = pd.DataFrame(
        data=[
            [
                input_item.hypothesis_id,
                input_item.premise,
                input_item.hypothesis_segment,
                inference_score.inference,
                inference_score.explanation,
            ]
            for input_item, inference_score in zip(input_batch, inference_scores)
        ],
        columns=pd.Series(
            [
                _InternalColumns.hypothesis_id,
                _InternalColumns.premise,
                _InternalColumns.hypothesis,
                _InternalColumns.nli,
                _InternalColumns.explanation,
            ]
        ),
    )
    df_pre_aggregation_scores[_InternalColumns.json] = df_pre_aggregation_scores.apply(
        lambda row: _row_to_json(row), axis=1
    )

    aggregated_inferences = df_pre_aggregation_scores.groupby(
        _InternalColumns.hypothesis_id
    )[_InternalColumns.nli].mean()
    aggregated_jsons = (
        df_pre_aggregation_scores.groupby(_InternalColumns.hypothesis_id)[
            _InternalColumns.json
        ]
        .apply(list)
        .apply(lambda x: json.dumps([json.loads(js) for js in x]))
    )
    highlights = pd.DataFrame(
        {
            _InternalColumns.json: aggregated_jsons,
            _InternalColumns.hypothesis_split: segmented_hypotheses,
        }
    ).apply(_json_to_highlight, axis=1)
    inference_returns = [
        InferenceReturn(inference=inference, json=js, highlight=highlight)
        for inference, js, highlight in zip(
            aggregated_inferences, aggregated_jsons, highlights
        )
    ]
    return inference_returns


def calculate_inference(
    premise: Premise,
    hypothesis: Hypothesis,
    llm: BaseChatModel,
    question: Optional[Question] = None,
    documents: Optional[Documents] = None,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> InferenceReturn:
    """
    Calculates the inference of a hypothesis from a premise.

    Parameters
    -----------

        premise : str
            The text of the premise from which the hypothesis will be inferred.

        hypothesis : str
            The text of the hypothesis.

        llm : BaseChatModel
            The Langchain chat model used for calculating inference.

        question : str, optional, default=None
            A question related to the inference process as a part of the premise.

        documents : List[str], optional, default=None
            A document names that used in the inference process  as a part of the premise.

        max_concurrency : int, default=8
            The maximum number of concurrent requests to the LLM.

        batch_size : int, default=6
            The maximum number of objects processed in a single prompt for simple tasks.

        show_progress_bar : bool, default=True
            Whether to display a progress bar during LLM requests.

    Returns
    ------------
    InferenceReturn
        Returns the inference,
        along with a JSON string that explains how the inference was derived and
        highlights string used for highlighting each segment of the hypothesis.
    """
    questions = None if question is None else [question]
    list_documents = None if documents is None else [documents]
    inference_returns = calculate_batch_inference(
        [premise],
        [hypothesis],
        llm,
        questions,
        list_documents,
        max_concurrency,
        batch_size,
        show_progress_bar,
    )
    return inference_returns[0]
