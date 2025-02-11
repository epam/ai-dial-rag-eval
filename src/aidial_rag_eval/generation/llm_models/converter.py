from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List, Tuple

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_core.utils.json import parse_json_markdown

from aidial_rag_eval.generation.prompts import no_pronouns_prompt
from aidial_rag_eval.generation.types import HypothesisSegment
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
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


def json_to_dict_returns(input_: AIMessage) -> Dict:
    try:
        return_dict = parse_json_markdown(str(input_.content))
        assert isinstance(return_dict, dict)
        return return_dict
    except OutputParserException:
        return {}


def dict_returns_to_segments(input_: Dict) -> List[HypothesisSegment]:
    try:
        return input_["sentences"]
    except (TypeError, KeyError):
        return []


BatchInfo = namedtuple("BatchInfo", ["hypothesis_id", "start_index"])


def segment_batch_with_info(
    hypothesis_id: int, segments: List[HypothesisSegment], batch_size: int
) -> Tuple[List[List[HypothesisSegment]], List[BatchInfo]]:
    segment_batches = []
    batch_infos = []

    for i in range((len(segments) - 1) // batch_size + 1):
        segment_batches.append(segments[i * batch_size : (i + 1) * batch_size + 1])
        batch_infos.append(BatchInfo(hypothesis_id, i * batch_size + 1))
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

    def __init__(self, model: BaseChatModel, batch_size=16, max_concurrency=32):

        self._chain = (
            no_pronouns_prompt
            | model
            | RunnableLambda(json_to_dict_returns)
            | RunnableLambda(dict_returns_to_segments)
        )
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ):
        original_segment_batches: List[List[HypothesisSegment]] = []
        batch_infos: List[BatchInfo] = []
        for hypothesis_id, segmented_text in enumerate(segmented_texts):
            segments = segmented_text.segments
            if len(segmented_text.segments) <= 1:
                continue
            batch, batch_info = segment_batch_with_info(
                hypothesis_id, segments, self.batch_size
            )
            original_segment_batches.extend(batch)
            batch_infos.extend(batch_info)

        no_pronouns_segment_batches = self._get_no_pronouns_segments(
            original_segment_batches, show_progress_bar
        )

        for batch_info, no_pronouns_segment_batch, original_segment_batch in zip(
            batch_infos, no_pronouns_segment_batches, original_segment_batches
        ):
            if len(no_pronouns_segment_batch) != len(original_segment_batch[1:]):
                continue
            segmented_texts[batch_info.hypothesis_id].replace_segments(
                no_pronouns_segment_batch,
                batch_info.start_index,
            )

    def _get_no_pronouns_segments(
        self,
        original_segment_batches: List[List[HypothesisSegment]],
        show_progress_bar: bool,
    ) -> List[List[HypothesisSegment]]:
        with ProgressBarCallback(
            len(original_segment_batches), show_progress_bar
        ) as cb:
            no_pronouns_segment_batches = self._chain.batch(
                [{"sentences": batch} for batch in original_segment_batches],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        return no_pronouns_segment_batches
