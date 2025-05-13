from dataclasses import dataclass
from enum import Enum
from typing import List, TypeVar

FactType = TypeVar("FactType")
ContextChunk = str
Document = str
Text = str

Context = List[ContextChunk]
Facts = List[FactType]
Documents = List[Document]
Question = Text
Answer = Text
GroundTruthAnswer = Text


@dataclass
class GroundTruth:
    question: Question
    documents: Documents
    facts: Facts
    answer: GroundTruthAnswer


@dataclass
class CollectedAnswers:
    question: Question
    documents: Documents
    context: Context
    answer: Answer


class GroundTruthColumns(str, Enum):
    QUESTION = "question"
    DOCUMENTS = "documents"
    FACTS = "facts"
    ANSWER = "answer"


class AnswerColumns(str, Enum):
    QUESTION = "question"
    DOCUMENTS = "documents"
    CONTEXT = "context"
    ANSWER = "answer"


class MergedColumns(str, Enum):
    QUESTION = GroundTruthColumns.QUESTION.value
    CONTEXT = AnswerColumns.CONTEXT.value
    ANSWER = AnswerColumns.ANSWER.value
    GROUND_TRUTH_ANSWER = "ground_truth_answer"
    DOCUMENTS = GroundTruthColumns.DOCUMENTS.value
    FACTS = GroundTruthColumns.FACTS.value
    JOINED_CONTEXT = "joined_context"


MERGED_KEY_COLUMNS = [MergedColumns.DOCUMENTS, MergedColumns.QUESTION]
ANSWERS_COLUMNS = [col.value for col in AnswerColumns if isinstance(col, AnswerColumns)]
GROUND_TRUTH_COLUMNS = [
    col.value for col in GroundTruthColumns if isinstance(col, GroundTruthColumns)
]
