from abc import ABC, abstractmethod
from typing import List

from aidial_rag_eval.generation.types import Result, Statement
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


class StatementExtractor(ABC):
    """
    Abstract base class for creating StatementExtractor.

    Input is a list of Hypothesis objects.
    """

    @abstractmethod
    def extract(
        self,
        segmented_hypotheses: List[Result[SegmentedText]],
        show_progress_bar: bool,
    ) -> List[Result[List[List[Statement]]]]:
        pass
