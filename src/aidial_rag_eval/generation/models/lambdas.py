from json import JSONDecodeError
from typing import Dict, List

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import chain
from langchain_core.utils.json import parse_json_markdown


@chain
def json_to_list(input_: AIMessage) -> List:
    """
    Intermediate part of the chain that converts the output of the LLM into a JSON,
    the outer part of which is a list.

    Parameters
    -----------
    input_: AIMessage
        Output from the LLM.

    Returns
    ------------
    List
        Returns a list; if the LLM output was incorrect, returns an empty list.
    """
    try:
        return_list = parse_json_markdown(str(input_.content))
        assert isinstance(return_list, list)
        return return_list
    except (OutputParserException, JSONDecodeError, AssertionError):
        return []


@chain
def json_to_dict(input_: AIMessage) -> Dict[str, List[str]]:
    """
    Intermediate part of the chain that converts the output of the LLM into a JSON,
    the outer part of which is a dict.

    Parameters
    -----------
    input_ : AIMessage
        The output from the LLM.

    Returns
    ------------
    Dict[str, List[str]]
        LLM output if valid;
        otherwise, an empty dict is returned.
    """
    try:
        return_dict = parse_json_markdown(str(input_.content))
        assert isinstance(return_dict, dict)
        return return_dict
    except (
        TypeError,
        OutputParserException,
        JSONDecodeError,
        AssertionError,
    ):
        return {}
