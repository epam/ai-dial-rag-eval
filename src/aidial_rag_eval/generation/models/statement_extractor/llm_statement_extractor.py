from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable, chain

from aidial_rag_eval.generation.models.lambdas import json_to_list
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
    statements_for_each_hypothesis: List[Dict[str, List[str]]],
) -> List[List[str]]:
    """
    Function is part of a chain that extracts segments from a list.

    Parameters
    -----------
    statements_for_each_hypothesis : List[Dict[str, List[str]]]
        The output list of dicts from the LLM with extracted statements.

    Returns
    ------------
    List[str]
        The extracted statements. if the LLM output is valid;
        otherwise, an empty list is returned.
    """
    try:
        return [
            return_dict["statements"] for return_dict in statements_for_each_hypothesis
        ]
    except (
        TypeError,
        KeyError,
    ):
        return []


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

        self._chain = statement_prompt | model | json_to_list | list_to_statements
        self.max_concurrency = max_concurrency

    def extract(
        self,
        hypothesis_segments: List[List[HypothesisSegment]],
        show_progress_bar: bool,
    ) -> List[List[List[Statement]]]:
        """
        Method that calls a chain to extract statements from each
        hypothesis segment.

        Parameters
        -----------
        hypothesis_segments : List[List[HypothesisSegment]]
            A list of hypothesis segments as a sources of statements.

        show_progress_bar : bool
            A flag that controls the display of a progress bar

        Returns
        ------------
        List[List[List[Statement]]]
            Returns the statements for each hypothesis segment.
        """

        with ProgressBarCallback(len(hypothesis_segments), show_progress_bar) as cb:
            returns = self._chain.batch(
                [
                    {
                        "hypotheses": hypotheses,
                    }
                    for hypotheses in hypothesis_segments
                ],
                config={"callbacks": [cb], "max_concurrency": self.max_concurrency},
            )
        return returns
