from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda


class FakeStructuredChatModel(BaseChatModel):
    responses: list[Any] = []
    side_effect: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("Use with_structured_output instead")

    def with_structured_output(self, schema, **kwargs) -> RunnableLambda:
        def get_response(input_value) -> Any:
            if self.side_effect is not None:
                raise self.side_effect
            resp = self.responses.pop(0)
            if isinstance(resp, tuple):
                expected_input, output = resp
                actual_input = input_value.to_string()
                assert actual_input == expected_input, (
                    f"Unexpected LLM input.\nExpected:"
                    f"\n{expected_input}\n\nActual:\n{actual_input}"
                )
                return output
            return resp

        return RunnableLambda(get_response)
