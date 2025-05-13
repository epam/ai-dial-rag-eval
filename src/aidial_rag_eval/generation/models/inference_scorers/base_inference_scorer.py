from abc import ABC, abstractmethod
from typing import List

from aidial_rag_eval.generation.types import InferenceInputs, InferenceScore


class InferenceScorer(ABC):
    """
    Abstract base class for creating InferenceScorer to calculate
    inference of a hypothesis from a premise.
    """

    @abstractmethod
    def get_inference(
        self,
        inference_inputs: List[InferenceInputs],
        show_progress_bar: bool,
    ) -> List[InferenceScore]:
        return [InferenceScore(inference=0.0, explanation="")] * len(inference_inputs)
