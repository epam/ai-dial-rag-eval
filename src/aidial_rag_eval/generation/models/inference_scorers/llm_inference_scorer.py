import json
from typing import List

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable, chain

from aidial_rag_eval.generation.models.inference_scorers.base_inference_scorer import (
    InferenceScorer,
)
from aidial_rag_eval.generation.models.inference_scorers.inference_template import (
    inference_prompt,
)
from aidial_rag_eval.generation.models.lambdas import json_to_returns
from aidial_rag_eval.generation.types import InferenceInputs, InferenceScore
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback


@chain
def returns_to_inference_score(input_: List) -> InferenceScore:
    """
    The final part of the chain for calculating inference.
    The input is a list of tagged facts from the hypothesis segment.
    The inference is the average proportion of "ENT" tags.

    Parameters
    -----------
    input_ : List[Dict]
        Output from the LLM, where the hypothesis segment is broken down into facts,
        tagged as "CONT", "NEUT", or "ENT", with an explanation provided.

    Returns
    ------------
    InferenceScore
        Returns the inference and an explanation of how the inference was obtained.
        If the LLM output is incorrect, the inference is 0.
    """
    try:
        list_tags = [d["tag"] for d in input_]
        inference = float(np.mean([tag == "ENT" for tag in list_tags]))
        assert not np.isnan(inference)
        explanation = json.dumps(input_)
    except (TypeError, KeyError, AssertionError):
        inference = 0.0
        explanation = ""
    return InferenceScore(inference=inference, explanation=explanation)


class LLMInferenceScorer(InferenceScorer):
    """
    The LLMInferenceScorer is designed to calculate
    inference of a hypothesis from a premise using a LLM.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and transformation of JSON into InferenceScore."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(
        self,
        model: BaseChatModel,
        max_concurrency: int,
    ):

        self._chain = (
            inference_prompt | model | json_to_returns | returns_to_inference_score
        )
        self.max_concurrency = max_concurrency

    def get_inference(
        self,
        inference_inputs: List[InferenceInputs],
        show_progress_bar: bool,
    ) -> List[InferenceScore]:
        """
        Function that calls a chain to calculate inference
        segments of hypotheses from a premise.

        Parameters
        -----------
        inference_inputs : List[InferenceInputs]
            A list of InferenceInputs, where each element includes a hypothesis_segment
            for which we want to calculate inference,
            a premise from which we are trying to derive the hypothesis_segment,
            and other additional information for the inference process.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[InferenceScore]
            Returns the inferences and additionally
            returns an explanation of how the inference was obtained
            for each input.
        """
        with ProgressBarCallback(len(inference_inputs), show_progress_bar) as cb:
            returns = self._chain.batch(
                [
                    {
                        "premise": batch_element.premise,
                        "hypothesis": batch_element.hypothesis_segment,
                        "document": batch_element.document_name,
                    }
                    for batch_element in inference_inputs
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        assert isinstance(returns, list)
        return returns
