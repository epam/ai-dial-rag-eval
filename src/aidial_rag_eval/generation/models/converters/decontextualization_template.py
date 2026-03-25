# flake8: noqa
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
    get_structured_output_instruction,
)


class DecontextualizationOutput(BaseModel):
    """The list must contain exactly as many segments as in the input, in the same order."""

    segments: List[str] = Field(
        description=(
            "Decontextualized segments, one per input segment, in the same order. "
            "If the input has N segments, this list must have exactly N items."
        )
    )


decontextualization_template = """
The task is to replace all pronouns in segments with their corresponding nouns or proper names when their referents are known.
You will receive segments.
If a segment is nonsensical, a reference, link, or meaningless, return it unchanged.
If unsure what to do with segment, return the original segment.
Only perform the task; do not shorten, simplify, or correct errors.
Do not provide explanations.

For example: "My mom is a good person.", "She always takes care of me."
should return segments: ["My mom is a good person.", "My mom always takes care of me."]

Important: the response must have the same number of segments, split the same way.

List of input segments:
{{ sentences_str }}
"""


def get_decontextualization_prompt(method: StructuredOutputMethod) -> PromptTemplate:
    return PromptTemplate.from_template(
        template=decontextualization_template
        + get_structured_output_instruction(method),
        template_format="jinja2",
    )
