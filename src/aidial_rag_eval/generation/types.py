from dataclasses import dataclass, fields
from typing import Union

from aidial_rag_eval.types import Answer, GroundTruthAnswer, Text

TextSegment = str
JoinedContext = Text

Premise = Union[JoinedContext, Answer, GroundTruthAnswer]
Hypothesis = Union[Answer, GroundTruthAnswer]
HypothesisSegment = TextSegment
JoinedDocumentsName = str

MetricBind = str


@dataclass
class InferenceBatchItem:
    hypothesis_id: int
    premise: Premise
    hypothesis_segment: HypothesisSegment
    document_name: JoinedDocumentsName


@dataclass
class InferenceReturn:
    inference: float
    json: str
    highlight: str


inference_column = fields(InferenceReturn)[0].name


@dataclass
class RefusalReturn:
    refusal: float
