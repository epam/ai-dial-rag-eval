# flake8: noqa
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


class DecontextualizationOutput(BaseModel):
    segments: List[str]


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

decontextualization_prompt = PromptTemplate.from_template(
    template=decontextualization_template,
    template_format="jinja2",
)
