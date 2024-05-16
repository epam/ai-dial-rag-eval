from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np

FactType = TypeVar("FactType")
ContextChunk = str

Context = list[ContextChunk]
Facts = list[FactType]


FactsRanks = np.ndarray
ContextRelevance = np.ndarray


class FactMatchResult(NamedTuple):
    facts_ranks: FactsRanks
    context_relevance: ContextRelevance


@runtime_checkable
class Matcher(Protocol[FactType]):
    @staticmethod
    def match_facts(  # noqa: E704
        facts: list[FactType], context: list[ContextChunk]
    ) -> FactMatchResult: ...


Documents = list[str]
Question = str


@dataclass
class GroundTruth:
    question: Question
    documents: Documents
    facts: Facts


@dataclass
class CollectedAnswers:
    question: Question
    documents: Documents
    context: Context
    answer: str
