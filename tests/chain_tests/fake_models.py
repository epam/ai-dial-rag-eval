from typing import List, Optional

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class FakeRecordingChatModel(FakeListChatModel):
    recorded_inputs: List[List[BaseMessage]] = []
    side_effect: Optional[Exception] = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.recorded_inputs.append(messages)
        if self.side_effect is not None:
            raise self.side_effect
        return super()._generate(messages, stop, run_manager, **kwargs)
