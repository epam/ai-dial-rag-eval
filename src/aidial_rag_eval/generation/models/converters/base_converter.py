from abc import ABC, abstractmethod
from typing import List

from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


class SegmentConverter(ABC):
    """
    Abstract base class for creating SegmentConverter.

    Input is a list of SegmentedText objects to convert.
    """

    @abstractmethod
    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ):
        pass
