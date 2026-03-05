import json
from json import JSONDecodeError
from typing import Dict, List

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableSerializable,
    chain,
)
from langchain_core.utils.json import parse_json_markdown

from aidial_rag_eval.generation.models.converters.base_converter import SegmentConverter
from aidial_rag_eval.generation.models.converters.decontextualization_template import (
    decontextualization_prompt,
)
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


@chain
def check_if_sentences_less_than_2(input_: Dict) -> bool:
    assert type(input_) is dict
    return len(input_["segmented_text"].segments) < 2


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


@chain
def segmented_text_to_json_list(input_: Dict) -> Dict:
    assert type(input_) is dict
    return {"sentences_str": json.dumps(input_["segmented_text"].segments)}


@chain
def return_original_segmented_text(input_: Dict) -> Dict:
    assert type(input_) is dict
    return input_["segmented_text"]


@chain
def dict_segments_to_segmented_text(llm_outputs_with_inputs: Dict) -> SegmentedText:
    original_segmented_text: SegmentedText = llm_outputs_with_inputs["segmented_text"]
    try:
        decontextualized_segments = llm_outputs_with_inputs["decontextualized_segments"]
        assert len(decontextualized_segments) == len(original_segmented_text.segments)
        return SegmentedText(
            decontextualized_segments, original_segmented_text.delimiters
        )
    except (TypeError, KeyError, AssertionError):
        return original_segmented_text


class LLMNoPronounsConverter(SegmentConverter):
    """
    Converter that decontextualizes text segments using an LLM.

    Takes a list of SegmentedText objects and processes each one:
    - If a SegmentedText has fewer than 2 segments, it is returned unchanged.
    - Otherwise, all segments are sent to the LLM for decontextualization.

    The LLM replaces pronouns and context-dependent references
    to make each segment self-contained.
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
        self._chain = RunnableBranch(
            (check_if_sentences_less_than_2, return_original_segmented_text),
            RunnablePassthrough.assign(
                decontextualized_segments=segmented_text_to_json_list
                | decontextualization_prompt
                | model
                | json_to_dict_segments
            )
            | dict_segments_to_segmented_text,
        )
        self.max_concurrency = max_concurrency

    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ) -> List[SegmentedText]:
        """
        Method that converts segmented texts by replacing pronouns using an LLM.
        The LLM processes all segments and returns converted segments.
        If the invariant of the length of input and output segment batches
        is not maintained, the segments of this batch are not replaced.

        Parameters
        -----------
        segmented_texts : List[SegmentedText]
            A list of segmented texts where segment replacement occurs.

        show_progress_bar : bool
            A flag that controls the display of a progress bar.

        Returns
        -------
        List[SegmentedText]
            A list of segmented texts with decontextualized segments.
        """
        with ProgressBarCallback(len(segmented_texts), show_progress_bar) as cb:
            decontextualized_segmented_texts = self._chain.batch(
                [
                    {
                        "segmented_text": segmented_text,
                    }
                    for segmented_text in segmented_texts
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        return decontextualized_segmented_texts
