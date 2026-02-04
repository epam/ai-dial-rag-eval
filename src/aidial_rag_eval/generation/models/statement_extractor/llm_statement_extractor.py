from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
    chain,
)

from aidial_rag_eval.generation.models.lambdas import json_to_list, safe_model_invoke
from aidial_rag_eval.generation.models.statement_extractor.base_statement_extractor import (
    StatementExtractor,
)
from aidial_rag_eval.generation.models.statement_extractor.statement_extractor_template import (
    statement_prompt,
)
from aidial_rag_eval.generation.types import HypothesisSegment, Statement
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback


@chain
def list_to_statements(
    llm_outputs_with_inputs: Dict,
) -> List[List[str]]:
    """
    Function is part of a chain that extracts segments from a list.

    Parameters
    -----------
    llm_outputs_with_inputs : Dict
        Passed inputs with output list of dicts from the LLM with extracted statements.

    Returns
    ------------
    List[List[str]]
        The extracted statements. if the LLM output is valid;
        otherwise, return an original list of hypothesis_segments,
        each hypothesis_segment is wrapped.
    """
    try:
        statements_for_hypothesis_segments = llm_outputs_with_inputs[
            "llm_output_statements"
        ]
        hypothesis_segments = llm_outputs_with_inputs["hypothesis_segments"]
        assert len(hypothesis_segments) == len(statements_for_hypothesis_segments)
        return [
            return_dict["statements"]
            for return_dict in llm_outputs_with_inputs["llm_output_statements"]
        ]
    except (
        TypeError,
        KeyError,
        AssertionError,
    ):
        return [
            [hypothesis_segment]
            for hypothesis_segment in llm_outputs_with_inputs["hypothesis_segments"]
        ]


@chain
def wrap_hypotheses(input_: Dict) -> Dict:
    assert type(input_) is dict
    return {
        "hypotheses": [
            f"<hypothesis{index + 1}> {hypothesis_segment} </hypothesis{index + 1}>"
            for index, hypothesis_segment in enumerate(input_["hypothesis_segments"])
        ],
    }


class LLMStatementExtractor(StatementExtractor):
    """
    The LLMStatementExtractor is designed to extract
    statements from a hypothesis segment using a LLM.
    """

    _chain: RunnableSerializable
    """A chain that contains the core logic, which includes:
    the prompt, model, conversion of output content to JSON,
    and transformation of JSON into statements."""

    max_concurrency: int
    """Configuration attribute for _chain.batch,
    indicating how many prompts will be sent in parallel."""

    def __init__(
        self,
        model: BaseChatModel,
        max_concurrency: int,
    ):

        self._chain = (
            RunnablePassthrough.assign(
                llm_output_statements=wrap_hypotheses
                | statement_prompt
                | RunnableLambda(lambda x: safe_model_invoke(model, x))
                | json_to_list
            )
            | list_to_statements
        )
        self.max_concurrency = max_concurrency

    def extract(
        self,
        list_of_hypothesis_segments: List[List[HypothesisSegment]],
        show_progress_bar: bool,
    ) -> List[List[List[Statement]]]:
        """
        Method that calls a chain to extract statements from each
        hypothesis segment.

        Parameters
        -----------
        list_of_hypothesis_segments : List[List[HypothesisSegment]]
            A list of hypothesis segments as a sources of statements.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[List[List[Statement]]]
            Returns the statements for each hypothesis segment.
        """

        with ProgressBarCallback(
            len(list_of_hypothesis_segments), show_progress_bar
        ) as cb:
            returns = self._chain.batch(
                [
                    {
                        "hypothesis_segments": hypothesis_segments,
                    }
                    for hypothesis_segments in list_of_hypothesis_segments
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        return returns
