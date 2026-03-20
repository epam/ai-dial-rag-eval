from typing import Dict, List, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

from aidial_rag_eval.generation.models.statement_extractor.base_statement_extractor import (
    StatementExtractor,
)
from aidial_rag_eval.generation.models.statement_extractor.statement_extractor_template import (
    StatementsOutput,
    statement_prompt,
)
from aidial_rag_eval.generation.types import ErrorInfo, HypothesisStatements
from aidial_rag_eval.generation.utils.exceptions import wrap_batch_errors
from aidial_rag_eval.generation.utils.progress_bar import ProgressBarCallback
from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


@chain
def segmented_text_result_to_dict(input_: SegmentedText) -> Dict:
    return {"hypothesis_segments": input_.segments}


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
    output: StatementsOutput = llm_outputs_with_inputs[
        "llm_output_statements"
    ]
    print("--------------------------------------------------------")
    print("input:", hypothesis_segments)
    print("output", [item.statements for item in output.hypothesis_statements])
    print("--------------------------------------------------------")
    assert len(hypothesis_segments) == len(output.hypothesis_statements), (
        f"Statement extraction LLM response"
        f" has {len(output.hypothesis_statements)} items,"
        f" expected {len(hypothesis_segments)}"
    )
    return [item.statements for item in output.hypothesis_statements]


def _make_statement_prompt_input(hypothesis_segments: list) -> Dict:
    return {
        "hypotheses": [
            f"<hypothesis{index + 1}> {hypothesis_segment} </hypothesis{index + 1}>"
            for index, hypothesis_segment in enumerate(hypothesis_segments)
        ],
    }


@chain
def wrap_hypotheses(input_: Dict) -> Dict:
    assert type(input_) is dict
    return _make_statement_prompt_input(input_["hypothesis_segments"])


class LLMStatementExtractor(StatementExtractor):
    """
    The LLMStatementExtractor is designed to extract
    statements from a hypothesis segment using a LLM.
    """

    _chain: Runnable
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
        @chain
        def statement_chain(input_: Union[SegmentedText, ErrorInfo]):
            if isinstance(input_, ErrorInfo):
                return input_
            return (
                segmented_text_result_to_dict
                | RunnablePassthrough.assign(
                    llm_output_statements=wrap_hypotheses
                    | statement_prompt
                    | model.with_structured_output(StatementsOutput)
                )
                | list_to_statements
            )

        self._chain = statement_chain
        self.max_concurrency = max_concurrency

    def extract(
        self,
        segmented_hypotheses: List[Union[SegmentedText, ErrorInfo]],
        show_progress_bar: bool,
    ) -> List[Union[HypothesisStatements, ErrorInfo]]:
        """
        Method that calls a chain to extract statements from each
        hypothesis segment.

        Parameters
        -----------
        segmented_hypotheses : List[Union[SegmentedText, ErrorInfo]]
            A list of segmented hypotheses or ErrorInfo.
            Errors are passed through without calling the LLM.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[Union[HypothesisStatements, ErrorInfo]]
            Returns the statements for each hypothesis segment,
            or errors if extraction failed for that item.
        """
        with ProgressBarCallback(len(segmented_hypotheses), show_progress_bar) as cb:
            raw_results = self._chain.batch(
                segmented_hypotheses,
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
                return_exceptions=True,
            )
        return wrap_batch_errors(raw_results)
