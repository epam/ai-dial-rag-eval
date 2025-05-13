from abc import ABC, abstractmethod
from typing import List

from aidial_rag_eval.generation.types import RefusalReturn


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
