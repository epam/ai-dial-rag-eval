import itertools
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable, chain
from more_itertools import chunked

from aidial_rag_eval.generation.models.lambdas import json_to_list
from aidial_rag_eval.generation.models.refusal_detectors.base_refusal_detector import (
    RefusalDetector,
)
from aidial_rag_eval.generation.models.refusal_detectors.refusal_template import (
    refusal_prompt,
)
from aidial_rag_eval.generation.types import RefusalReturn
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.types import Answer


@chain
def returns_to_refusal_return(input_: List) -> List[RefusalReturn]:
    """
    The final part of the chain, which calculates answer refusals
    for each answer in the batch based on the JSON output from the LLM.

    Parameters
    -----------
    input_: List
        Output from the LLM, where each batch element is tagged with "REJ"
        if the answer is an answer refusal, or "ANS" otherwise.

    Returns
    ------------
    List[RefusalReturn]
        Returns a list of RefusalReturn, where each input answer from the batch
        is assigned a 1. if it is an answer refusal, or 0. otherwise.
    """
    return [RefusalReturn(refusal=float(tag == "REJ")) for tag in input_]


class LLMRefusalDetector(RefusalDetector):
    """
    Experimental:
    The LLMRefusalDetector is designed to calculate
    answer refusal using a LLM.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and  and transformation of JSON into RefusalReturn."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(
        self,
        model: BaseChatModel,
        max_concurrency: int,
    ):

        self._chain = refusal_prompt | model | json_to_list | returns_to_refusal_return
        self.max_concurrency = max_concurrency

    def get_refusal(
        self, answers: List[Answer], show_progress_bar: bool
    ) -> List[RefusalReturn]:
        """
        Method that calls a chain to calculate answer refusals
        for each answer.

        Parameters
        -----------
        answers: List[Answer] : List[InferenceInputs]
            A list of answers or ground truth answers.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[RefusalReturn]
            Returns the answer refusal for each answer.
        """
        batches = list(chunked(answers, 10))

        with ProgressBarCallback(len(batches), show_progress_bar) as cb:
            refusal_returns = self._chain.batch(
                [
                    {
                        "answers": hypotheses,
                    }
                    for hypotheses in batches
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        refusal_returns = [
            (
                refusal_return
                if len(refusal_return) == len(input_batch)
                else [RefusalReturn(refusal=0.0)] * len(input_batch)
            )
            for refusal_return, input_batch in zip(refusal_returns, batches)
        ]
        return list(itertools.chain.from_iterable(refusal_returns))
