from abc import ABC, abstractmethod
from itertools import chain
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from more_itertools import chunked

from aidial_rag_eval.generation.llm_models.lambdas import json_to_returns
from aidial_rag_eval.generation.prompts import refusal_prompt
from aidial_rag_eval.generation.types import RefusalReturn
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.types import Answer


class RefusalDetector(ABC):
    """
    Abstract base class for creating RefusalDetector to calculate
    answer refusal.
    """

    @abstractmethod
    def get_refusal(
        self, answers: List[str], show_progress_bar: bool
    ) -> List[RefusalReturn]:
        return [RefusalReturn(refusal=0.0)] * len(answers)


def returns_to_refusal_return(input_: List) -> List[RefusalReturn]:
    return [RefusalReturn(refusal=float(tag == "REJ")) for tag in input_]


class LLMRefusalDetector(RefusalDetector):
    """
    The LLMRefusalDetector is designed to calculate
    answer refusal using a LLM.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and  and transformation of JSON into RefusalReturn."""

    batch_size: int
    """The number of answers that will be processed simultaneously in the _chain."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(self, model: BaseChatModel, batch_size: int, max_concurrency: int):

        self._chain = (
            refusal_prompt
            | model
            | RunnableLambda(json_to_returns)
            | RunnableLambda(returns_to_refusal_return)
        )
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    def get_refusal(
        self, answers: List[Answer], show_progress_bar: bool
    ) -> List[RefusalReturn]:
        batches = list(chunked(answers, self.batch_size))

        with ProgressBarCallback(len(batches), show_progress_bar) as cb:
            refusal_returns = self._chain.batch(
                [{"answers": hypotheses} for hypotheses in batches],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        refusal_returns = [
            (
                refusal_return
                if len(refusal_return) == len(batch)
                else [RefusalReturn(refusal=0.0)] * len(batch)
            )
            for refusal_return, batch in zip(refusal_returns, batches)
        ]
        return list(chain.from_iterable(refusal_returns))
