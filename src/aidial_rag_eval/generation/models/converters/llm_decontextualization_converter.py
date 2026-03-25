import json
from typing import Dict, List, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

from aidial_rag_eval.generation.models.converters.base_converter import SegmentConverter
from aidial_rag_eval.generation.models.converters.decontextualization_template import (
    DecontextualizationOutput,
    get_decontextualization_prompt,
)
from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
)
from aidial_rag_eval.generation.types import ErrorInfo
from aidial_rag_eval.generation.utils.exceptions import wrap_batch_errors
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


@chain
def segmented_text_to_json_list(input_: Dict) -> Dict:
    assert type(input_) is dict
    return {"sentences_str": json.dumps(input_["segmented_text"].segments)}


@chain
def return_original_segmented_text(input_: Dict) -> SegmentedText:
    assert type(input_) is dict
    return input_["segmented_text"]


@chain
def dict_segments_to_segmented_text(llm_outputs_with_inputs: Dict) -> SegmentedText:
    original_segmented_text: SegmentedText = llm_outputs_with_inputs["segmented_text"]
    output: DecontextualizationOutput = llm_outputs_with_inputs[
        "decontextualized_segments"
    ]
    assert len(output.segments) == len(original_segmented_text.segments), (
        f"Decontextualization LLM response has {len(output.segments)} segments,"
        f" expected {len(original_segmented_text.segments)}"
    )
    return SegmentedText(output.segments, original_segmented_text.delimiters)


class LLMNoPronounsConverter(SegmentConverter):
    """
    Converter that decontextualizes text segments using an LLM.

    Takes a list of SegmentedText objects and processes each one:
    - If a SegmentedText has fewer than 2 segments, it is returned unchanged.
    - Otherwise, all segments are sent to the LLM for decontextualization.

    The LLM replaces pronouns and context-dependent references
    to make each segment self-contained.
    """

    _chain: Runnable
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
        structured_output_method: StructuredOutputMethod = "function_calling",
    ):
        structured_model = model.with_structured_output(
            DecontextualizationOutput, method=structured_output_method
        )
        prompt = get_decontextualization_prompt(structured_output_method)

        @chain
        def pronouns_converter_chain(input_: Dict):
            if len(input_["segmented_text"].segments) < 2:
                return return_original_segmented_text
            return (
                RunnablePassthrough.assign(
                    decontextualized_segments=segmented_text_to_json_list
                    | prompt
                    | structured_model
                )
                | dict_segments_to_segmented_text
            )

        self._chain = pronouns_converter_chain
        self.max_concurrency = max_concurrency

    def transform_texts(
        self, segmented_texts: List[SegmentedText], show_progress_bar: bool
    ) -> List[Union[SegmentedText, ErrorInfo]]:
        """
        Method that converts segmented texts by replacing pronouns using an LLM.

        Parameters
        -----------
        segmented_texts : List[SegmentedText]
            A list of segmented texts where segment replacement occurs.

        show_progress_bar : bool
            A flag that controls the display of a progress bar.

        Returns
        -------
        List[Union[SegmentedText, ErrorInfo]]
            A list where each element is either a decontextualized SegmentedText
            wrapped in Result, or a Result with error set if processing failed.
        """
        with ProgressBarCallback(len(segmented_texts), show_progress_bar) as cb:
            raw_results = self._chain.batch(
                [
                    {"segmented_text": segmented_text}
                    for segmented_text in segmented_texts
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
                return_exceptions=True,
            )
        return wrap_batch_errors(raw_results)
