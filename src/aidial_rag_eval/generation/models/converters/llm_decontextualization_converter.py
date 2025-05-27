from json import JSONDecodeError
from typing import List

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable, chain
from langchain_core.utils.json import parse_json_markdown

from aidial_rag_eval.generation.models.converters.base_converter import SegmentConverter
from aidial_rag_eval.generation.models.converters.decontextualization_template import (
    decontextualization_prompt,
)
from aidial_rag_eval.generation.types import TextSegment
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


@chain
def json_to_dict_segments(input_: AIMessage) -> List[str]:
    """
    Function is part of a chain that extracts segments from an AIMessage.

    Parameters
    -----------
    input_ : AIMessage
        The output from the LLM which includes content with transformed segments.

    Returns
    ------------
    List[str]
        The transformed segments if the LLM output is valid;
        otherwise, an empty list is returned.
    """
    try:
        return_dict = parse_json_markdown(str(input_.content))
        assert isinstance(return_dict, dict)
        return return_dict["segments"]
    except (
        TypeError,
        KeyError,
        OutputParserException,
        JSONDecodeError,
        AssertionError,
    ):
        return []


class LLMNoPronounsConverter(SegmentConverter):
    """
    The LLMNoPronounsBatchConverter is designed to replace pronouns
    in text segments using a LLM.

    Input is a list of SegmentedText objects.
    If a SegmentedText object contains more than one segment,
    segments are sent in a prompt to the LLM.
    In a prompt, the first segment is used only for context,
    and pronoun replacement is performed only in the remaining segments.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and extraction of segments from JSON."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(
        self,
        model: BaseChatModel,
        max_concurrency: int,
    ):

        self._chain = decontextualization_prompt | model | json_to_dict_segments
        self.max_concurrency = max_concurrency

    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ):
        """
        Method that converts segmented texts by replacing pronouns using an LLM.
        The LLM processes segments,
        where the additional first segment is not converted
        but is provided for context to enable the conversion of the second sentence.
        The LLM returns converted segments.
        If the invariant of the length of input and output segment batches
        is not maintained, the segments of this batch are not replaced.

        Parameters
        -----------
        segmented_texts : List[SegmentedText]
            A list of segmented texts where segment replacement occurs.

        show_progress_bar : bool
            A flag that controls the display of a progress bar.
        """
        original_segment_batches: List[List[TextSegment]] = []
        segment_ids: List[int] = []
        for text_id, segmented_text in enumerate(segmented_texts):
            segments = segmented_text.segments
            if len(segments) <= 1:
                continue
            original_segment_batches.append(segments)
            segment_ids.append(text_id)

        no_pronouns_segment_batches = self._get_no_pronouns_segments(
            original_segment_batches, show_progress_bar
        )

        for text_id, no_pronouns_segment_batch, original_segment_batch in zip(
            segment_ids, no_pronouns_segment_batches, original_segment_batches
        ):
            if len(no_pronouns_segment_batch) != len(original_segment_batch):
                continue
            segmented_texts[text_id].replace_segments(
                no_pronouns_segment_batch[1:],
                1,
            )

    def _get_no_pronouns_segments(
        self,
        original_segment_batches: List[List[TextSegment]],
        show_progress_bar: bool,
    ) -> List[List[TextSegment]]:
        """
        Method that calls _chain to replace pronouns.

        Parameters
        -----------
        original_segment_batches : List[List[str]]
            Segments of texts.

        show_progress_bar : bool
            A flag that controls the display of a progress bar.
        Returns
        ------------
        List[List[str]]
            List of converted segments, divided into batches.
        """
        with ProgressBarCallback(
            len(original_segment_batches), show_progress_bar
        ) as cb:
            no_pronouns_segment_batches = self._chain.batch(
                [
                    {
                        "sentences": batch,
                    }
                    for batch in original_segment_batches
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        return no_pronouns_segment_batches
