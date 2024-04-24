from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

FactsRanks = np.ndarray
ContextRelevance = np.ndarray

FactMatchResult = tuple[FactsRanks, ContextRelevance]


FactType = TypeVar("FactType")
ContextChunk = str

Context = list[ContextChunk]
Facts = list[FactType]


class Matcher(Generic[FactType]):
    @staticmethod
    def match_facts(
        facts: list[FactType], context: list[ContextChunk]
    ) -> FactMatchResult:
        raise NotImplementedError


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
