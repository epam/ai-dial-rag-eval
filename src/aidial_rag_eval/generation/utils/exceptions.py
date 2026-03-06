import traceback


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))
