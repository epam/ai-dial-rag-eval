import itertools
import json
from typing import Iterable, List, Optional, Tuple, TypeVar

import numpy as np
from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.models.converters.llm_decontextualization_converter import (
    LLMNoPronounsConverter,
)
from aidial_rag_eval.generation.models.inference_scorers.llm_inference_scorer import (
    LLMInferenceScorer,
)
from aidial_rag_eval.generation.models.statement_extractor.llm_statement_extractor import (
    LLMStatementExtractor,
)
from aidial_rag_eval.generation.types import (
    Hypothesis,
    HypothesisSegment,
    InferenceInputs,
    InferenceReturn,
    InferenceScore,
    JoinedDocumentsName,
    Premise,
    Statement,
)
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from aidial_rag_eval.types import Documents, Question


def _join_documents(documents: Documents) -> JoinedDocumentsName:
    return " ; ".join(documents)


def _make_inference_task_inputs(
    premises: List[Premise],
    statements: List[List[List[Statement]]],
    document_names: List[JoinedDocumentsName],
) -> List[InferenceInputs]:
    """
    The function collects input data for the inference task.

    Parameters
    -----------
    premises : List[str]
        A list of premises from which we want to derive hypotheses in pairs.

    statements : List[List[List[Statement]]]
        A deeply nested list of statements, where the outermost list corresponds to
        different hypotheses, the next level represents the segmentation of each
        hypothesis into hypothesis segments, and the innermost list breaks each
        hypothesis segment down into individual statements.

    document_names: List[str]
        A list of document names used as additional information for the inference task.

    Returns
    ------------
    List[InferenceInputs]
        A list that has as many items as there are innermost lists of statements,
        each innermost statement list is paired with the premise, document names
        and the ID of its origin hypothesis.
    """
    inference_inputs = list(
        itertools.chain.from_iterable(
            [
                [
                    InferenceInputs(
                        hypothesis_id=i,
                        premise=premises[i],
                        statements=list_statements,
                        document_name=document_names[i],
                    )
                    for list_statements in statements[i]
                ]
                for i in range(len(statements))
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
    segmented_text: SegmentedText,
) -> str:
    """
    Function that aggregates the inference results of segments
    for the same hypothesis in JSON format.

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
        JSON string of the inference for the hypothesis.
    """
    return json.dumps(
        [
            {
                "inference": inference_score.inference,
                "hypothesis": segment,
                "premise": [inference_input.premise],
                "explanation": inference_score.explanation,
            }
            for (inference_input, inference_score), segment in zip(
                grouped_data_item, segmented_text.segments
            )
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
    for (_, inference_score), segment, delimiter in zip(
        grouped_data_item, segmented_text.segments, segmented_text.delimiters + [""]
    ):
        highlight["corpus"].append(
            {
                "text": segment,
                "score": inference_score.inference - 1,
                "title": inference_score.inference,
            }
        )
        highlight["corpus"].append({"text": delimiter, "score": 0.0})
    return json.dumps(highlight)


def segment_hypotheses(
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[SegmentedText]:
    """
    Function that segments hypotheses into hypothesis segments(roughly into
    sentences), and then removes pronouns using LLM.

    Parameters
    -----------

    hypotheses : List[str]
        The text of the hypothesis.

    llm : BaseChatModel
        The Langchain chat model used for calculating inference.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    show_progress_bar : bool, default=True
        Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[SegmentedText]
        List of hypothesis segments with delimiters.
    """
    converter = LLMNoPronounsConverter(
        model=llm,
        max_concurrency=max_concurrency,
    )

    segmented_hypotheses = [
        SegmentedText.from_text(text=hypothesis) for hypothesis in hypotheses
    ]
    if show_progress_bar:
        print("Converting hypothesis...")
    converter.transform_texts(segmented_hypotheses, show_progress_bar)
    return segmented_hypotheses


def extract_statements(
    hypotheses_segments: List[List[HypothesisSegment]],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[List[List[Statement]]]:
    """
    Function that extracts statements from each hypothesis segment.
    Hypothesis segments of the inner list are grouped together and
    fed into the prompt.

    Parameters
    -----------

    hypotheses_segments : List[str]
        List of hypothesis segments.

    llm : BaseChatModel
        The Langchain chat model used for calculating inference.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    show_progress_bar : bool, default=True
        Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[List[List[Statement]]]
        A deeply nested list of statements, where the outermost list corresponds to
        different hypotheses, the next level corresponds to the segmentation of each
        hypothesis into hypothesis segments, and the innermost list breaks each
        hypothesis segment down into individual statements.
    """
    extractor = LLMStatementExtractor(
        model=llm,
        max_concurrency=max_concurrency,
    )
    if show_progress_bar:
        print("Extracting statements...")
    statements = extractor.extract(
        hypotheses_segments,
        show_progress_bar,
    )
    return statements


def infer_statements(
    premises: List[Premise],
    statements: List[List[List[Statement]]],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[List[Tuple[InferenceInputs, InferenceScore]]]:
    """
    Function that infers statements.
    Statements of the innermost list are grouped together and
    fed into the prompt.

    Parameters
    -----------

    premises : List[str]
        The text of the premise from which the hypothesis will be inferred.

    statements : List[List[List[Statement]]]
        A deeply nested list of statements, where the outermost list corresponds to
        different hypotheses, the next level corresponds to the segmentation of each
        hypothesis into hypothesis segments, and the innermost list breaks each
        hypothesis segment down into individual statements.

    llm : BaseChatModel
        The Langchain chat model used for calculating inference.

    questions : List[str], optional, default=None
        A questions related to the inference process as a part of the premise.

    list_documents : List[List[str]], optional, default=None
        A list of document names that used
        in the inference process as a part of the premises.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    show_progress_bar : bool, default=True
        Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[List[Tuple[InferenceInputs, InferenceScore]]]
        A nested list of inputs and outputs of the inference step grouped by hypothesis
        segment.
    """
    scorer = LLMInferenceScorer(
        model=llm,
        max_concurrency=max_concurrency,
    )
    if list_documents is None:
        document_names: List[JoinedDocumentsName] = [""] * len(premises)
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
        statements,
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
    return list(map(_iterable_group_with_key_to_list_group, iterable_groups_with_id))


def calculate_batch_inference(
    premises: List[Premise],
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
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

    show_progress_bar : bool, default=True
        Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[InferenceReturn]
        Returns the list of inference,
        along with a JSON strings that explains how the inference was derived and
        highlights strings used for highlighting each segment of each hypothesis.
    """

    segmented_hypotheses: List[SegmentedText] = segment_hypotheses(
        hypotheses=hypotheses,
        llm=llm,
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
    )
    statements: List[List[List[Statement]]] = extract_statements(
        hypotheses_segments=[
            segmented_hypothesis.segments
            for segmented_hypothesis in segmented_hypotheses
        ],
        llm=llm,
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
    )
    grouped_data_list: List[List[Tuple[InferenceInputs, InferenceScore]]] = (
        infer_statements(
            premises=premises,
            statements=statements,
            llm=llm,
            questions=questions,
            list_documents=list_documents,
            max_concurrency=max_concurrency,
            show_progress_bar=show_progress_bar,
        )
    )

    aggregated_inferences = map(
        lambda grouped_data_item: float(
            np.mean(
                [inference_score.inference for _, inference_score in grouped_data_item]
            )
        ),
        grouped_data_list,
    )

    aggregated_jsons = itertools.starmap(
        _grouped_data_item_to_json,
        zip(grouped_data_list, segmented_hypotheses),
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
        show_progress_bar=show_progress_bar,
    )
    return inference_returns[0]
