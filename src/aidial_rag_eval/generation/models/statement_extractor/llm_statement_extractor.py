from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableSerializable,
    chain,
)

from aidial_rag_eval.generation.models.lambdas import json_to_list, wrap_in_result
from aidial_rag_eval.generation.models.statement_extractor.base_statement_extractor import (
    StatementExtractor,
)
from aidial_rag_eval.generation.models.statement_extractor.statement_extractor_template import (
    statement_prompt,
)
from aidial_rag_eval.generation.types import Result, Statement
from aidial_rag_eval.generation.utils.exceptions import format_exception
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


@chain
def check_if_error_present(input_: Result[SegmentedText]) -> bool:
    return bool(input_.error)


@chain
def return_error_as_statement_result(input_: Result[SegmentedText]) -> Result:
    return Result(error=input_.error)


@chain
def segmented_text_result_to_dict(input_: Result[SegmentedText]) -> Dict:
    assert input_.value is not None
    return {"hypothesis_segments": input_.value.segments}


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
        The extracted statements if the LLM output is valid.
    """
    hypothesis_segments = llm_outputs_with_inputs["hypothesis_segments"]
    statements_for_hypothesis_segments = llm_outputs_with_inputs[
        "llm_output_statements"
    ]
    assert len(hypothesis_segments) == len(statements_for_hypothesis_segments)
    return [
        return_dict["statements"] for return_dict in statements_for_hypothesis_segments
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
        self._chain = RunnableBranch(
            (check_if_error_present, return_error_as_statement_result),
            segmented_text_result_to_dict
            | RunnablePassthrough.assign(
                llm_output_statements=wrap_hypotheses
                | statement_prompt
                | model
                | json_to_list
            )
            | list_to_statements
            | wrap_in_result,
        )
        self.max_concurrency = max_concurrency

    def extract(
        self,
        segmented_hypotheses: List[Result[SegmentedText]],
        show_progress_bar: bool,
    ) -> List[Result[List[List[Statement]]]]:
        """
        Method that calls a chain to extract statements from each
        hypothesis segment.

        Parameters
        -----------
        segmented_hypotheses : List[Result[SegmentedText]]
            A list of segmented hypotheses wrapped in Result.
            Items with error set are passed through without calling the LLM.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[Result[List[List[Statement]]]]
            Returns the statements for each hypothesis segment wrapped in Result,
            or a Result with error set if extraction failed for that item.
        """
        with ProgressBarCallback(len(segmented_hypotheses), show_progress_bar) as cb:
            raw_results = self._chain.batch(
                segmented_hypotheses,
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
                return_exceptions=True,
            )
        return [
            (
                result
                if isinstance(result, Result)
                else Result(error=format_exception(result))
            )
            for result in raw_results
        ]
