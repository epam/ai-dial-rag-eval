import dataclasses
import json
from dataclasses import dataclass, fields
from typing import List, Optional, Union

from aidial_rag_eval.types import Answer, GroundTruthAnswer, Text

TextSegment = str
JoinedContext = Text

Premise = Union[JoinedContext, Answer, GroundTruthAnswer]
Hypothesis = Union[Answer, GroundTruthAnswer]
Statement = str
HypothesisSegmentStatements = List[Statement]
HypothesisStatements = List[HypothesisSegmentStatements]
JoinedDocumentsName = str

MetricBind = str


@dataclass
class ErrorInfo:
    """Error information from a failed chain step.

    name: exception class name, e.g. "ValueError"
    traceback: full formatted traceback string
    """

    error_repr: str
    traceback: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


@dataclass
class InferenceInputs:
    """Input data used for calculating inference"""

    hypothesis_id: int
    premise: Premise
    statements: List[Statement]
    document_name: JoinedDocumentsName
    error: Optional[ErrorInfo] = None


@dataclass
class InferenceScore:
    """Inference score for a hypothesis segment, calculated based on InferenceInputs"""

    inference: float
    explanation: str
    error: Optional[ErrorInfo] = None


@dataclass
class InferenceReturn:
    """Inference for a hypothesis, aggregated results for hypothesis segments"""

    inference: float
    inference_min: float
    inference_max: float
    json: str
    highlight: str
    errors: List[Optional[str]]


# Used for calculating mean and median inferences
inference_column = fields(InferenceReturn)[0].name


@dataclass
class RefusalReturn:
    """Answer refusal calculated for the answer"""

    refusal: float
    refusal_error: Optional[str] = None
