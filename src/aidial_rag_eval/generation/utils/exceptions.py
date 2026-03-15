import traceback

from aidial_rag_eval.generation.types import ErrorInfo


def make_error_info(exc: BaseException) -> ErrorInfo:
    return ErrorInfo(
        error_repr=repr(exc),
        traceback="".join(traceback.format_exception(exc)),
    )
