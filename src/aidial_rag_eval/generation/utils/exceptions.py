import traceback

from aidial_rag_eval.generation.types import ErrorInfo


def make_error_info(exc: BaseException) -> ErrorInfo:
    return ErrorInfo(
        name=type(exc).__name__,
        traceback="".join(traceback.format_exception(exc)),
    )
