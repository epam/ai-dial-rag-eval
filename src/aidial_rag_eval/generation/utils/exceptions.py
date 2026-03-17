import traceback
from typing import Callable, List, TypeVar, Union

from aidial_rag_eval.generation.types import ErrorInfo

T = TypeVar("T")
E = TypeVar("E")


def make_error_info(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        error_repr=repr(exc),
        traceback="".join(traceback.format_exception(exc)),
    )


def wrap_batch_errors(
    raw_results: List[Union[T, Exception]],
    make_error: Callable[[Exception], E] = make_error_info,
) -> List[Union[T, E]]:
    return [
        (result if not isinstance(result, Exception) else make_error(result))
        for result in raw_results
    ]
