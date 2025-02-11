from typing import List

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.utils.json import parse_json_markdown


def json_to_returns(input_: AIMessage) -> List:
    try:
        return_list = parse_json_markdown(str(input_.content))
        assert isinstance(return_list, list)
        return return_list
    except OutputParserException:
        return []
