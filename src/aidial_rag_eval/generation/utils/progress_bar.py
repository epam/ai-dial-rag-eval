from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from tqdm.autonotebook import tqdm


class ProgressBarCallback(BaseCallbackHandler):
    def __init__(self, total: int, show_progress_bar: bool):
        super().__init__()
        self.progress_bar = tqdm(total=total, disable=not show_progress_bar)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.progress_bar.update(1)

    def __enter__(self):
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__(exc_type, exc_value, exc_traceback)

    def __del__(self):
        self.progress_bar.__del__()
