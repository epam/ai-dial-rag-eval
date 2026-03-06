from dataclasses import dataclass, fields
from typing import Generic, List, Optional, TypeVar, Union

from aidial_rag_eval.types import Answer, GroundTruthAnswer, Text

TextSegment = str
JoinedContext = Text

Premise = Union[JoinedContext, Answer, GroundTruthAnswer]
Hypothesis = Union[Answer, GroundTruthAnswer]
Statement = str
JoinedDocumentsName = str

MetricBind = str

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Wrapper for a chain output that may have failed.

    Either value is set (success) or error is set (failure).
    The error field contains a formatted traceback string.
    """

    value: Optional[T] = None
    error: Optional[str] = None


@dataclass
class InferenceInputs:
    """Input data used for calculating inference"""

    hypothesis_id: int
    premise: Premise
    statements: List[Statement]
    document_name: JoinedDocumentsName
    error: Optional[str] = None


@dataclass
class InferenceScore:
    """Inference score for a hypothesis segment, calculated based on InferenceInputs"""

    inference: Optional[float]
    explanation: str
    error: Optional[str] = None


@dataclass
class InferenceReturn:
    """Inference for a hypothesis, aggregated results for hypothesis segments"""

    inference: Optional[float]
    inference_min: float
    inference_max: float
    json: str
    highlight: str


# Used for calculating mean and median inferences
inference_column = fields(InferenceReturn)[0].name


@dataclass
class RefusalReturn:
    """Answer refusal calculated for the answer"""

    refusal: Optional[float]
    refusal_error: Optional[str] = None
