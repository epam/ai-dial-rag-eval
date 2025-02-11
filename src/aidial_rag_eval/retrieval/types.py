from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from aidial_rag_eval.types import ContextChunk, FactType

FactsRanks = np.ndarray
ContextRelevance = np.ndarray
ContextHighlight = npt.NDArray[np.str_]


class FactMatchResult(NamedTuple):
    facts_ranks: FactsRanks
    context_relevance: ContextRelevance
    context_highlight: ContextHighlight


@runtime_checkable
class Matcher(Protocol[FactType]):
    @staticmethod
    def match_facts(  # noqa: E704
        facts: list[FactType], context: list[ContextChunk]
    ) -> FactMatchResult: ...
