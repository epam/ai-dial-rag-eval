from collections import namedtuple
from json import JSONDecodeError
from typing import List, Tuple

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


BatchInfo = namedtuple("BatchInfo", ["text_id", "start_index"])


def segment_batch_with_info(
    text_id: int, segments: List[TextSegment], batch_size: int
) -> Tuple[List[List[TextSegment]], List[BatchInfo]]:
    """
    Function that splits segments into batches while preserving metadata.

    Parameters
    -----------
    text_id : int
        The id of the text to be split into batches.

    segments : List[str]
        A list of segments into which the text has been split.

    batch_size : int
        The maximum size of a batch.

    Returns
    ------------
    Tuple[List[List[str]], List[BatchInfo]]
        Returns the segments split into batches along
        with the corresponding metadata for each batch.
        For each batch, the following information is preserved:
        - text_id: the id from which the batch is derived
        - start_index: the start index of the batch in the list of segments.
    """
    segment_batches = []
    batch_infos = []

    for i in range((len(segments) - 1) // batch_size + 1):
        segment_batches.append(segments[i * batch_size : (i + 1) * batch_size + 1])
        batch_infos.append(BatchInfo(text_id, i * batch_size + 1))
    return segment_batches, batch_infos


class LLMNoPronounsBatchConverter(SegmentConverter):
    """
    The LLMNoPronounsBatchConverter is designed to replace pronouns
    in text segments using a LLM.

    Input is a list of SegmentedText objects.
    If a SegmentedText object contains more than one segment,
    a maximum of batch_size + 1 segments are sent in a single prompt to the LLM.
    In a single prompt, the first segment is used only for context,
    and pronoun replacement is performed only in the remaining segments.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and extraction of segments from JSON."""

    batch_size: int
    """Specifies the number of segments that the _chain will process simultaneously,
    which is batch_size + 1 (an additional segment is needed for context).
    The _chain will return batch_size segments,
    processing all sentences except the first one."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(
        self,
        model: BaseChatModel,
        batch_size: int,
        max_concurrency: int,
    ):

        self._chain = decontextualization_prompt | model | json_to_dict_segments
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ):
        """
        Function that converts segmented texts by replacing pronouns using an LLM.
        Input segments are divided into batches
        while preserving metadata about their origin.
        The LLM processes a maximum of batch_size + 1 segments,
        where the additional first segment is not converted
        but is provided for context to enable the conversion of the second sentence.
        The LLM returns a maximum of batch_size converted segments;
        the first segment is not returned.
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
        batch_infos: List[BatchInfo] = []
        for text_id, segmented_text in enumerate(segmented_texts):
            segments = segmented_text.segments
            if len(segmented_text.segments) <= 1:
                continue
            batch, batch_info = segment_batch_with_info(
                text_id, segments, self.batch_size
            )
            original_segment_batches.extend(batch)
            batch_infos.extend(batch_info)

        no_pronouns_segment_batches = self._get_no_pronouns_segments(
            original_segment_batches, show_progress_bar
        )

        for batch_info, no_pronouns_segment_batch, original_segment_batch in zip(
            batch_infos, no_pronouns_segment_batches, original_segment_batches
        ):
            if len(no_pronouns_segment_batch) != len(original_segment_batch):
                continue
            segmented_texts[batch_info.text_id].replace_segments(
                no_pronouns_segment_batch[1:],
                batch_info.start_index,
            )

    def _get_no_pronouns_segments(
        self,
        original_segment_batches: List[List[TextSegment]],
        show_progress_bar: bool,
    ) -> List[List[TextSegment]]:
        """
        Function that calls _chain to replace pronouns.

        Parameters
        -----------
        original_segment_batches : List[List[str]]
            Segments of texts, divided into batches.

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
