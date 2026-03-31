import json
import math
from typing import Dict, List

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

from aidial_rag_eval.generation.models.inference_scorers.base_inference_scorer import (
    InferenceScorer,
)
from aidial_rag_eval.generation.models.inference_scorers.inference_template import (
    StatementInferenceOutput,
    get_inference_prompt,
)
from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
)
from aidial_rag_eval.generation.types import ErrorInfo, InferenceInputs, InferenceScore
from aidial_rag_eval.generation.utils.exceptions import (
    make_error_info,
    wrap_batch_errors,
)
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback


@chain
def returns_to_inference_score(llm_outputs_with_inputs: Dict) -> InferenceScore:
    """
    The final part of the chain for calculating inference.
    The inference is the average proportion of "ENT" tags among the possible tags:
    "ENT", "NEUT" and "CONT".

    Parameters
    -----------
    llm_outputs_with_inputs : Dict
        Passed inputs with a StatementInferenceOutput from the LLM
        stored in the "inference" key.

    Returns
    ------------
    InferenceScore
        Returns the inference and an explanation of how the inference was obtained.
        If the LLM output is incorrect, the inference is 0.
    """
    output: StatementInferenceOutput = llm_outputs_with_inputs["inference"]
    passed_statements = llm_outputs_with_inputs["statements"]
    assert len(output.statement_inference) == len(passed_statements), (
        f"Inference LLM response has {len(output.statement_inference)} outputs,"
        f" expected {len(passed_statements)}"
    )
    list_tags = [item.tag for item in output.statement_inference]
    inference = float(np.mean([tag == "ENT" for tag in list_tags]))
    assert not math.isnan(inference), "Inference LLM response produced NaN inference"
    explanation = json.dumps(
        [
            {"explanation": item.explanation, "tag": item.tag, "statement": s}
            for item, s in zip(output.statement_inference, passed_statements)
        ]
    )
    return InferenceScore(inference=inference, explanation=explanation)


@chain
def inference_inputs_to_dict(input_: InferenceInputs) -> Dict:
    return {
        "premise": input_.premise,
        "statements": input_.statements,
        "document": input_.document_name.strip(),
    }


def _make_inference_prompt_input(premise: str, statements: list, document: str) -> Dict:
    return {
        "premise": premise,
        "statements_json": json.dumps(statements, ensure_ascii=False),
        "document": document,
    }


@chain
def wrap_statements(input_: Dict) -> Dict:
    assert type(input_) is dict
    return _make_inference_prompt_input(
        premise=input_["premise"],
        statements=input_["statements"],
        document=input_["document"],
    )


class LLMInferenceScorer(InferenceScorer):
    """
    The LLMInferenceScorer is designed to calculate
    inference of a hypothesis from a premise using a LLM.
    """

    _chain: Runnable
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
        structured_output_method: StructuredOutputMethod = "function_calling",
    ):
        structured_model = model.with_structured_output(
            StatementInferenceOutput, method=structured_output_method
        )
        prompt = get_inference_prompt(structured_output_method)

        @chain
        def inference_chain(input_: InferenceInputs):
            if isinstance(input_.error, ErrorInfo):
                return InferenceScore(
                    inference=math.nan, explanation="", error=input_.error
                )
            if not input_.statements:
                return InferenceScore(inference=0.0, explanation="")
            return (
                inference_inputs_to_dict
                | RunnablePassthrough.assign(
                    inference=wrap_statements | prompt | structured_model
                )
                | returns_to_inference_score
            )

        self._chain = inference_chain
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
            returns an explanation of how the inference was obtained
            for each input.
        """
        with ProgressBarCallback(len(inference_inputs), show_progress_bar) as cb:
            raw_results = self._chain.batch(
                inference_inputs,
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
                return_exceptions=True,
            )
        assert isinstance(raw_results, list)
        return wrap_batch_errors(
            raw_results,
            lambda error: InferenceScore(
                inference=math.nan, explanation="", error=make_error_info(error)
            ),
        )
