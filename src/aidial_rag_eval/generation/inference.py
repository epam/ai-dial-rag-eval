import itertools
import json
from typing import Iterable, List, Optional, Tuple, TypeVar, Union, cast

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
    ErrorInfo,
    Hypothesis,
    InferenceInputs,
    InferenceReturn,
    InferenceScore,
    JoinedDocumentsName,
    Premise,
    Statement,
)
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from aidial_rag_eval.types import Documents, Question

T = TypeVar("T")


def _join_documents(documents: Documents) -> JoinedDocumentsName:
    return " ; ".join(documents)


def _make_inference_task_inputs(
    premises: List[Premise],
    statements: List[Union[List[List[Statement]], ErrorInfo]],
    document_names: List[JoinedDocumentsName],
) -> List[InferenceInputs]:
    """
    The function collects input data for the inference task.

    Parameters
    -----------
    premises : List[str]
        A list of premises from which we want to derive hypotheses in pairs.

    statements : List[Union[List[List[Statement]], ErrorInfo]]
        A deeply nested list of statements, where the outermost
        list corresponds to different hypotheses, the next level represents the
        segmentation of each hypothesis into hypothesis segments, and the innermost
        list breaks each hypothesis segment down into individual statements.
        If an error, a single placeholder InferenceInputs is created
        with the error propagated.

    document_names: List[str]
        A list of document names used as additional information for the inference task.

    Returns
    ------------
    List[InferenceInputs]
        A list that has as many items as there are innermost lists of statements.
        Each innermost statement list is paired with the premise, document name,
        and the ID of its origin hypothesis. For failed hypotheses, a single
        placeholder InferenceInputs with the error field set is created.
    """
    return list(
        itertools.chain.from_iterable(
            (
                [
                    InferenceInputs(
                        hypothesis_id=i,
                        premise=premises[i],
                        statements=list_statements,
                        document_name=document_names[i],
                    )
                    for list_statements in statement_result
                ]
                if not isinstance(statement_result, ErrorInfo)
                else [
                    InferenceInputs(
                        hypothesis_id=i,
                        premise=premises[i],
                        statements=[],
                        document_name=document_names[i],
                        error=statement_result,
                    )
                ]
            )
            for i, statement_result in enumerate(statements)
        )
    )


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
                "error": (
                    inference_score.error.to_json() if inference_score.error else None
                ),
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
                "score": (
                    (inference_score.inference - 1)
                    if inference_score.inference is not None
                    else None
                ),
                "title": inference_score.inference,
            }
        )
        highlight["corpus"].append({"text": delimiter, "score": 0.0})
    return json.dumps(highlight)


def _segment_hypotheses(
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
<<<<<<< feat/error-handling
) -> List[Union[SegmentedText, ErrorInfo]]:
=======
    auto_download_nltk: bool = True,
) -> List[SegmentedText]:
>>>>>>> development
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
    List[Union[SegmentedText, ErrorInfo]]
        List of decontextualized hypothesis segments,
        or error if processing failed for that item.
    """
    converter = LLMNoPronounsConverter(
        model=llm,
        max_concurrency=max_concurrency,
    )
    segmented_hypotheses = [
        SegmentedText.from_text(text=hypothesis, auto_download_nltk=auto_download_nltk)
        for hypothesis in hypotheses
    ]
    if show_progress_bar:
        print("Converting hypothesis...")
    return converter.transform_texts(segmented_hypotheses, show_progress_bar)


def _extract_statements(
    segmented_hypotheses: List[Union[SegmentedText, ErrorInfo]],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[Union[List[List[Statement]], ErrorInfo]]:
    """
    Function that extracts statements from each hypothesis segment.
    Hypothesis segments of the inner list are grouped together and
    fed into the prompt. Items that already have an error are passed through.

    Parameters
    -----------

    segmented_hypotheses : List[Union[SegmentedText, ErrorInfo]]
        Segmented hypotheses. Errors are
        passed through without calling the LLM.

    llm : BaseChatModel
        The Langchain chat model used for calculating inference.

    max_concurrency : int, default=8
        The maximum number of concurrent requests to the LLM.

    show_progress_bar : bool, default=True
        Whether to display a progress bar during LLM requests.

    Returns
    ------------
    List[Union[List[List[Statement]], ErrorInfo]]
        A deeply nested list of statements, where the outermost
        list corresponds to different hypotheses, the next level corresponds to the
        segmentation of each hypothesis into hypothesis segments, and the innermost
        list breaks each hypothesis segment down into individual statements.
        Error is returned for items where extraction failed.
    """
    extractor = LLMStatementExtractor(
        model=llm,
        max_concurrency=max_concurrency,
    )
    if show_progress_bar:
        print("Extracting statements...")
    return extractor.extract(segmented_hypotheses, show_progress_bar)


def _infer_statements(
    premises: List[Premise],
    statements: List[Union[List[List[Statement]], ErrorInfo]],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
    auto_download_nltk: bool = True,
) -> List[List[Tuple[InferenceInputs, InferenceScore]]]:
    """
    Function that infers statements.
    Statements of the innermost list are grouped together and
    fed into the prompt. Items that already have an error are passed through
    as placeholder InferenceInputs with the error propagated.

    Parameters
    -----------

    premises : List[str]
        The text of the premise from which the hypothesis will be inferred.

    statements : List[Union[List[List[Statement]], ErrorInfo]]
        A deeply nested list of statements, where the outermost
        list corresponds to different hypotheses, the next level corresponds to the
        segmentation of each hypothesis into hypothesis segments, and the innermost
        list breaks each hypothesis segment down into individual statements.
        Errors are for items that already failed in a previous stage.

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
        A nested list of inputs and outputs of the inference step grouped by hypothesis.
        For items that failed in any stage, InferenceScore will have inference=None
        and the error field set.
    """
    adjusted_premises: List[Premise] = list(premises)
    if questions is not None:
<<<<<<< feat/error-handling
        for i, question in enumerate(questions):
            question_split = SegmentedText.from_text(text=question)
            adjusted_premises[i] = question_split.segments[-1] + "\n" + premises[i]

    document_names: List[JoinedDocumentsName] = (
        [""] * len(statements)
        if list_documents is None
        else [_join_documents(docs) for docs in list_documents]
    )

=======
        segmented_questions = [
            SegmentedText.from_text(
                text=question, auto_download_nltk=auto_download_nltk
            )
            for question in questions
        ]
        premises = [
            question_split.segments[-1] + "\n" + premise
            for question_split, premise in zip(segmented_questions, premises)
        ]
>>>>>>> development
    inference_inputs = _make_inference_task_inputs(
        adjusted_premises,
        statements,
        document_names,
    )
    scorer = LLMInferenceScorer(
        model=llm,
        max_concurrency=max_concurrency,
    )
    if show_progress_bar:
        print("Getting inference...")
    inference_scores = scorer.get_inference(inference_inputs, show_progress_bar)

    iterable_groups_with_id = itertools.groupby(
        zip(inference_inputs, inference_scores), lambda x: x[0].hypothesis_id
    )
    return list(map(_iterable_group_with_key_to_list_group, iterable_groups_with_id))


def segment_hypotheses(
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[Optional[SegmentedText]]:
    return [
        None if isinstance(r, ErrorInfo) else r
        for r in _segment_hypotheses(
            hypotheses, llm, max_concurrency, show_progress_bar
        )
    ]


def extract_statements(
    segmented_hypotheses: List[SegmentedText],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[Optional[List[List[Statement]]]]:
    return [
        None if isinstance(r, ErrorInfo) else r
        for r in _extract_statements(
            cast(List[Union[SegmentedText, ErrorInfo]], segmented_hypotheses),
            llm,
            max_concurrency,
            show_progress_bar,
        )
    ]


def infer_statements(
    premises: List[Premise],
    statements: List[List[List[Statement]]],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
) -> List[List[Tuple[InferenceInputs, InferenceScore]]]:
    return _infer_statements(
        premises,
        cast(List[Union[List[List[Statement]], ErrorInfo]], statements),
        llm,
        questions,
        list_documents,
        max_concurrency,
        show_progress_bar,
    )


def calculate_batch_inference(
    premises: List[Premise],
    hypotheses: List[Hypothesis],
    llm: BaseChatModel,
    questions: Optional[List[Question]] = None,
    list_documents: Optional[List[Documents]] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
    auto_download_nltk: bool = True,
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

    segmented_hypotheses: List[Union[SegmentedText, ErrorInfo]] = _segment_hypotheses(
        hypotheses=hypotheses,
        llm=llm,
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
        auto_download_nltk=auto_download_nltk,
    )
    statements: List[Union[List[List[Statement]], ErrorInfo]] = _extract_statements(
        segmented_hypotheses=segmented_hypotheses,
        llm=llm,
        max_concurrency=max_concurrency,
        show_progress_bar=show_progress_bar,
    )
    grouped_data_list: List[List[Tuple[InferenceInputs, InferenceScore]]] = (
        _infer_statements(
            premises=premises,
            statements=statements,
            llm=llm,
            questions=questions,
            list_documents=list_documents,
            max_concurrency=max_concurrency,
            show_progress_bar=show_progress_bar,
            auto_download_nltk=auto_download_nltk,
        )
    )

    inference_returns: List[InferenceReturn] = []
    for hypothesis_index, grouped_data_item in enumerate(grouped_data_list):
        segmented_text = segmented_hypotheses[hypothesis_index]
        assert not isinstance(segmented_text, ErrorInfo)
        inferences = [score.inference for _, score in grouped_data_item]
        errors = [
            score.error.to_json() if score.error else None
            for _, score in grouped_data_item
        ]
        mean_inference = (
            None
            if any(inference is None for inference in inferences)
            else float(np.mean(cast(List[float], inferences)))
        )
        # fill Nones with 0.0
        min_possible_inference = float(
            np.nan_to_num(np.array(inferences, dtype=float), nan=0.0).mean()
        )
        # fill Nones with 1.0
        max_possible_inference = float(
            np.nan_to_num(np.array(inferences, dtype=float), nan=1.0).mean()
        )
        inference_returns.append(
            InferenceReturn(
                inference=mean_inference,
                inference_min=min_possible_inference,
                inference_max=max_possible_inference,
                json=_grouped_data_item_to_json(grouped_data_item, segmented_text),
                highlight=_grouped_data_item_to_highlight(
                    grouped_data_item, segmented_text
                ),
                errors=errors,
            )
        )
    return inference_returns


def calculate_inference(
    premise: Premise,
    hypothesis: Hypothesis,
    llm: BaseChatModel,
    question: Optional[Question] = None,
    documents: Optional[Documents] = None,
    max_concurrency: int = 8,
    show_progress_bar: bool = True,
    auto_download_nltk: bool = True,
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
        auto_download_nltk=auto_download_nltk,
    )
    return inference_returns[0]
