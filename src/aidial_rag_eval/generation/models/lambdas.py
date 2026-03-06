from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import chain
from langchain_core.utils.json import parse_json_markdown

from aidial_rag_eval.generation.types import Result


@chain
def wrap_in_result(value: Any) -> Result:
    return Result(value=value)


@chain
def json_to_list(input_: AIMessage) -> List:
    """
    Intermediate part of the chain converts the output of the LLM into a JSON,
    where the outer structure is a dictionary containing a list.

    Parameters
    -----------
    input_: AIMessage
        Output from the LLM.

    Returns
    ------------
    List
        Returns a list; if the LLM output was incorrect, returns an empty list.
    """
    return_dict = parse_json_markdown(str(input_.content))
    assert isinstance(return_dict, dict)
    return_list = return_dict[list(return_dict.keys())[0]]
    assert isinstance(return_list, list)
    return return_list


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
    return_dict = parse_json_markdown(str(input_.content))
    assert isinstance(return_dict, dict)
    return return_dict
