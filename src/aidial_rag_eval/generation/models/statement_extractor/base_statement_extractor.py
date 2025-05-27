from abc import ABC, abstractmethod
from typing import List

from aidial_rag_eval.generation.types import HypothesisSegment, Statement


class StatementExtractor(ABC):
    """
    Abstract base class for creating StatementExtractor.

    Input is a list of Hypothesis objects.
    """

    @abstractmethod
    def extract(
        self,
        hypothesis_segments: List[List[HypothesisSegment]],
        show_progress_bar: bool,
    ) -> List[List[List[Statement]]]:
        pass
