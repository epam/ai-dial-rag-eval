from abc import ABC, abstractmethod
from typing import List, Union

from aidial_rag_eval.generation.types import ErrorInfo, Statement
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


class StatementExtractor(ABC):
    """
    Abstract base class for creating StatementExtractor.

    Input is a list of Hypothesis objects.
    """

    @abstractmethod
    def extract(
        self,
        segmented_hypotheses: List[Union[SegmentedText, ErrorInfo]],
        show_progress_bar: bool,
    ) -> List[Union[List[List[Statement]], ErrorInfo]]:
        raise NotImplementedError()
