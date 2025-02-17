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
class InferenceInputs:
    """Input data used for calculating inference"""

    hypothesis_id: int
    premise: Premise
    hypothesis_segment: HypothesisSegment
    document_name: JoinedDocumentsName


@dataclass
class InferenceScore:
    """Inference score for a hypothesis segment, calculated based on InferenceInputs"""

    inference: float
    explanation: str


@dataclass
class InferenceReturn:
    """Inference for a hypothesis, aggregated results for hypothesis segments"""

    inference: float
    json: str
    highlight: str


# Used for calculating mean and median inferences
inference_column = fields(InferenceReturn)[0].name


@dataclass
class RefusalReturn:
    """Answer refusal calculated for the answer"""

    refusal: float
