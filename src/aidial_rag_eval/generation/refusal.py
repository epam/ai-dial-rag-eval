from typing import List

from langchain_core.language_models import BaseChatModel

from aidial_rag_eval.generation.models.refusal_detectors.llm_refusal_detector import (
    LLMRefusalDetector,
)
from aidial_rag_eval.generation.types import RefusalReturn
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText
from aidial_rag_eval.types import Answer


def calculate_batch_refusal(
    answers: List[Answer],
    llm: BaseChatModel,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> List[RefusalReturn]:
    """
    Checks if the answers are answer refusal.

    Parameters
    -----------

        answers : List[str]
            The list of the answers.

        llm : BaseChatModel
            The Langchain chat model used for calculating inference.

        max_concurrency : int, default=8
            The maximum number of concurrent requests to the LLM.

        batch_size : int, default=6
            The maximum number of objects processed in a single prompt for simple tasks.

        show_progress_bar : bool, default=True
            Whether to display a progress bar during LLM requests.

    Returns
    ------------
    RefusalReturn
        Returns the list of the answer refusals.
    """
    detector = LLMRefusalDetector(llm, batch_size, max_concurrency)
    answers_split = [SegmentedText.from_text(text=answer) for answer in answers]
    # As a heuristic, we send only the first 3 segments in the prompt.
    # We believe that if there are 3 whole segments with information
    # that is not related to refusal to answer,
    # we will not consider such a response as a refusal to answer
    # in any case.
    first_answers_sentences = [
        answers_split[i].get_joined_segments_by_range(0, 3)
        for i in range(len(answers_split))
    ]
    if show_progress_bar:
        print("Getting refusal...")
    return detector.get_refusal(first_answers_sentences, show_progress_bar)


def calculate_refusal(
    answer: Answer,
    llm: BaseChatModel,
    max_concurrency: int = 8,
    batch_size: int = 6,
    show_progress_bar: bool = True,
) -> RefusalReturn:
    """
    Checks if the answer is answer refusal.

    Parameters
    -----------

        answer : str
            The text of the answer.

        llm : BaseChatModel
            The Langchain chat model used for calculating inference.

        max_concurrency : int, default=8
            The maximum number of concurrent requests to the LLM.

        batch_size : int, default=6
            The maximum number of objects processed in a single prompt for simple tasks.

        show_progress_bar : bool, default=True
            Whether to display a progress bar during LLM requests.

    Returns
    ------------
    RefusalReturn
        Returns the answer refusal.
    """
    refusal_returns = calculate_batch_refusal(
        answers=[answer],
        llm=llm,
        max_concurrency=max_concurrency,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return refusal_returns[0]
