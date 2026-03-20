import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import Runnable, chain
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


class FakeStructuredChatModel(FakeListChatModel):
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs) -> Runnable:
        tool_name = convert_to_openai_tool(schema)["function"]["name"]

        @chain
        def content_to_tool_call(message: BaseMessage) -> BaseMessage:
            assert isinstance(message.content, str)
            args = json.loads(message.content)
            return AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": "fake_id"}],
            )

        return self | content_to_tool_call | PydanticToolsParser(tools=[schema], first_tool_only=True)