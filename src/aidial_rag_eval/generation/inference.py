import itertools
import json
from itertools import chain
from typing import Iterable, List, Optional, Tuple, TypeVar

import numpy as np
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsBatchConverter,
)
from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
)
from aidial_rag_eval.generation.types import (
    Hypothesis,
    InferenceInputs,
    InferenceReturn,
    InferenceScore,
    JoinedDocumentsName,
    Premise,
)
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from aidial_rag_eval.types import Documents, Question


def _join_documents(documents: Documents) -> JoinedDocumentsName:
    return " ; ".join(documents)


def _make_inference_task_inputs(
    premises: List[Premise],
    segmented_hypotheses: List[SegmentedText],
    document_names: List[JoinedDocumentsName],
) -> List[InferenceInputs]:
    """
    The function collects input data for the inference task.

    Parameters
    -----------
    premises : List[str]
        A list of premises from which we want to derive hypotheses in pairs.

    segmented_hypotheses : List[SegmentedText]
        A list of segmented hypotheses, where each segment is matched with its
        corresponding premise and a list of documents that correspond to the entire hypothesis.

    document_names: List[str]
        A list of document names used as additional information for the inference task.

    Returns
    ------------
    List[InferenceInputs]
        A list with a length equal to the total number of all segments from all hypotheses,
        where each hypothesis segment is matched with its corresponding premise document names,
        and the ID of the hypothesis from which it was taken.
    """
    inference_inputs = list(
        chain.from_iterable(
            [
                [
                    InferenceInputs(
                        hypothesis_id=i,
                        premise=premises[i],
                        hypothesis_segment=hypothesis,
                        document_name=document_names[i],
                    )
                    for hypothesis in segmented_hypotheses[i].segments
                ]
                for i in range(len(segmented_hypotheses))
            ]
        )
    )
    return inference_inputs


T = TypeVar("T")


def _iterable_group_with_key_to_list_group(
    iterable_group_with_key: Tuple[int, Iterable[T]],
) -> List[T]:
    """
    Function that transforms one of the groups obtained from itertools.groupby
    into a more convenient format:
    1) Removes the key used for groupby
    2) Converts the Iterable iterator into a List, preserving the internal objects.

    Parameters
    -----------
    iterable_group_with_key : Tuple[int, Iterable[Any]]
        A group from the results of itertools.groupby.

    Returns
    ------------
    List[Any]
        The same input group, but without the key and in List format.
    """
    return [pair for pair in iterable_group_with_key[1]]


def _grouped_data_item_to_json(
    grouped_data_item: List[Tuple[InferenceInputs, InferenceScore]],
) -> str:
    """
    Function that aggregates the inference results of segments
    for the same hypothesis in JSON format.

    Parameters
    -----------
    grouped_data_item : List[Tuple[InferenceInputs, InferenceScore]]
        Inference results of segments for the same hypothesis.

    Returns
    ------------
    str
        JSON string of the inference for the hypothesis.
    """
    return json.dumps(
        [
            {
                "inference": inference_score.inference,
                "hypothesis": inference_input.hypothesis_segment,
                "premise": [inference_input.premise],
                "explanation": inference_score.explanation,
            }
            for inference_input, inference_score in grouped_data_item
        ]
    )


def _grouped_data_item_to_highlight(
    grouped_data_item: List[Tuple[InferenceInputs, InferenceScore]],
    segmented_text: SegmentedText,
) -> str:
    """
    Function that converts inference results of segments from the same
    hypothesis into a JSON format for text highlighting.

    Parameters
    -----------
    grouped_data_item : List[Tuple[InferenceInputs, InferenceScore]]
        Inference results of segments for the same hypothesis.

    segmented_text : SegmentedText
        Segmented hypothesis containing both segments and delimiters
        for reconstructing the original text.

    Returns
    ------------
    str
        JSON string of highlights intended for coloring segments of the hypothesis.
    """
    highlight = {"corpus": []}
    for (inference_input, inference_score), delimiter in zip(
        grouped_data_item, segmented_text.delimiters + [""]
    ):
        highlight["corpus"].append(
            {
                "text": inference_input.hypothesis_segment,
                "score": inference_score.inference - 1,
                "title": inference_score.inference,
            }
        )
        highlight["corpus"].append({"text": delimiter, "score": 0.0})
    return json.dumps(highlight)


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
        highlights strings used for highlighting each segment of each hypothesis.
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
        document_names: List[JoinedDocumentsName] = [""] * len(hypotheses)
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
    inference_inputs = _make_inference_task_inputs(
        premises,
        segmented_hypotheses,
        document_names,
    )
    if show_progress_bar:
        print("Getting inference...")
    inference_scores = scorer.get_inference(
        inference_inputs,
        show_progress_bar,
    )

    iterable_groups_with_id = itertools.groupby(
        zip(inference_inputs, inference_scores), lambda x: x[0].hypothesis_id
    )
    grouped_data_list: List[List[Tuple[InferenceInputs, InferenceScore]]] = list(
        map(_iterable_group_with_key_to_list_group, iterable_groups_with_id)
    )

    aggregated_inferences = map(
        lambda grouped_data_item: float(
            np.mean(
                [inference_score.inference for _, inference_score in grouped_data_item]
            )
        ),
        grouped_data_list,
    )

    aggregated_jsons = map(
        _grouped_data_item_to_json,
        grouped_data_list,
    )
    highlights = itertools.starmap(
        _grouped_data_item_to_highlight, zip(grouped_data_list, segmented_hypotheses)
    )
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
        premises=[premise],
        hypotheses=[hypothesis],
        llm=llm,
        questions=questions,
        list_documents=list_documents,
        max_concurrency=max_concurrency,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return inference_returns[0]
