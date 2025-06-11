import json
from typing import Dict, List

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableSerializable,
    chain, RunnableParallel,
)

from aidial_rag_eval.generation.models.inference_scorers.base_inference_scorer import (
    InferenceScorer,
)
from aidial_rag_eval.generation.models.inference_scorers.inference_template import (
    inference_prompt, neutral_contradiction_prompt, entailment_prompt
)
from aidial_rag_eval.generation.models.lambdas import json_to_list
from aidial_rag_eval.generation.types import InferenceInputs, InferenceScore
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback


@chain
def returns_to_inference_score(llm_outputs_with_inputs: Dict) -> InferenceScore:
    """
    The final part of the chain for calculating inference.
    The inference is the average proportion of "ENT" tags among the possible tags:
    "Entailed", "Neutral" and "Contradiction".

    Parameters
    -----------
    llm_outputs_with_inputs : Dict
        Passed inputs with a list of tags and reasonings for each input statement
        stored in the "inference" key.

    Returns
    ------------
    InferenceScore
        Returns the inference and an reasoning of how the inference was obtained.
        If the LLM output is incorrect, the inference is 0.
    """
    try:
        outputs = llm_outputs_with_inputs["inference"]
        passed_statements = llm_outputs_with_inputs["statements"]
        neutral_contradiction_arguments = llm_outputs_with_inputs["neutral_contradiction_arguments"]
        entailment_arguments = llm_outputs_with_inputs["entailment_arguments"]
        list_tags = [d["tag"] for d in outputs]
        inference = float(np.mean([tag == "Entailment" for tag in list_tags]))
        assert len(outputs) == len(passed_statements)
        for d, s, nc_a, e_a in zip(
                outputs,
                passed_statements,
                neutral_contradiction_arguments,
                entailment_arguments
        ):
            d["statement"] = s
            d["neutral_contradiction_arguments"] = nc_a
            d["entailment_arguments"] = e_a
        assert not np.isnan(inference)
        reasoning = json.dumps(outputs)
    except (TypeError, KeyError, AssertionError):
        inference = 0.0
        reasoning = ""
    return InferenceScore(inference=inference, reasoning=reasoning)


@chain
def check_if_statements_is_empty(input_: Dict):
    assert type(input_) is dict
    return not input_.get("statements")


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
        self._chain = RunnableBranch(
            (
                check_if_statements_is_empty,
                lambda _: InferenceScore(inference=0.0, reasoning=""),
            ),
            RunnableParallel(
                neutral_contradiction_arguments=neutral_contradiction_prompt | model | JsonOutputParser(),
                entailment_arguments=entailment_prompt | model | JsonOutputParser(),
                premise=lambda x: x["premise"],
                statements=lambda x: x["statements"],
                document=lambda x: x["document"]
            ) | RunnablePassthrough.assign(
                inference=inference_prompt | model | json_to_list
            )
            | returns_to_inference_score,
        )
        self.max_concurrency = max_concurrency

    def get_inference(
        self,
        inference_inputs: List[InferenceInputs],
        show_progress_bar: bool,
    ) -> List[InferenceScore]:
        """
        Method that calls a chain to calculate inference
        of statements from a premise.

        Parameters
        -----------
        inference_inputs : List[InferenceInputs]
            A list of InferenceInputs, where each element includes statements
            for which we want to calculate inference,
            a premise from which we are trying to derive the statements,
            and other additional information for the inference process.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[InferenceScore]
            Returns the inferences and additionally
            returns an reasoning of how the inference was obtained
            for each input.
        """
        with ProgressBarCallback(len(inference_inputs), show_progress_bar) as cb:
            returns = self._chain.batch(
                [
                    {
                        "premise": batch_element.premise,
                        "statements": batch_element.statements,
                        "document": batch_element.document_name,
                    }
                    for batch_element in inference_inputs
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        assert isinstance(returns, list)
        return returns
