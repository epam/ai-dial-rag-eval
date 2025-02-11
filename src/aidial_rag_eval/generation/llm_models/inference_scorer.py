import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from aidial_rag_eval.generation.llm_models.lambdas import json_to_returns
from aidial_rag_eval.generation.prompts import nli_prompt
from aidial_rag_eval.generation.types import InferenceBatchItem
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback


@dataclass
class InferenceScore:
    inference: float
    explanation: str


class InferenceScorer(ABC):
    """
    Abstract base class for creating InferenceScorer to calculate
    inference of a hypothesis from a premise.
    """

    @abstractmethod
    def get_inference(
        self,
        batch: List[InferenceBatchItem],
        show_progress_bar: bool,
    ) -> List[InferenceScore]:
        return [InferenceScore(inference=0.0, explanation="")] * len(batch)


def returns_to_inference_score(input_: List) -> InferenceScore:
    try:
        list_tags = [d["tag"] for d in input_]
        inference = float(np.mean([tag == "ENT" for tag in list_tags]))
        explanation = json.dumps(input_)
    except (TypeError, KeyError):
        inference = 0.0
        explanation = ""
    return InferenceScore(inference=inference, explanation=explanation)


class LLMInferenceScorer(InferenceScorer):
    """
    The LLMInferenceScorer is designed to calculate
    inference of a hypothesis from a premise using a LLM.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and transformation of JSON into InferenceScore."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(self, model: BaseChatModel, max_concurrency: int):

        self._chain = (
            nli_prompt
            | model
            | RunnableLambda(json_to_returns)
            | RunnableLambda(returns_to_inference_score)
        )
        self.max_concurrency = max_concurrency

    def get_inference(
        self,
        batch: List[InferenceBatchItem],
        show_progress_bar: bool,
    ) -> List[InferenceScore]:
        with ProgressBarCallback(len(batch), show_progress_bar) as cb:
            returns = self._chain.batch(
                [
                    {
                        "premise": batch_element.premise,
                        "hypothesis": batch_element.hypothesis_segment,
                        "document": batch_element.document_name,
                    }
                    for batch_element in batch
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        assert isinstance(returns, list)
        return returns
