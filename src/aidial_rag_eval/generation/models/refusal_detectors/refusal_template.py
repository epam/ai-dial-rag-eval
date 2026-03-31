# flake8: noqa
from typing import List, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
    get_structured_output_instruction,
)


class RefusalTagsOutput(BaseModel):
    """The list must contain exactly as many tags as there are input answers. The first tag corresponds to answer1, the second to answer2, and so on."""

    tags: List[Literal["REJ", "ANS"]] = Field(
        description=(
            "One tag per input answer, in the same order. "
            'Use "REJ" if the answer is a refusal, "ANS" otherwise.'
        )
    )


refusal_template = """
Answer Refusal task is to determine if an answer should be tagged as a refusal to answer based on specific criteria.
The criteria include identifying if the answer explicitly states something is wrong with the question, request, or premise, or if it indicates a lack of information, irrelevance, or refusal to answer.

Single words, signs, numbers, links, etc. are not considered refusal to answer.

Lack of information can be formulated in different ways, pay attention to the list of synonyms.
Synonymous series:
1) premise, context, document, information.
2) hypothesis, answer.

Tagging guidelines:
- Use "REJ" if the answer is answer refusal, else tag it "ANS".

Example:
An explicit statement: "There is no answer to this question." should be tagged "REJ".
A statement that is not explicit: "The answer to this question is yes." should be tagged "ANS".

Each answer from the list of answers corresponds to a tag in your response.
The first answer corresponds to the first tag, the second corresponds to the second.
The number of tags must be the same as the number of answers in the answer list.
Each answer, even meaningless, must have it's own tag, if you don't know how to tag it, than just leave "ANS" tag for it.

Request:
List of answers:
{{ answers_json }}"""


def get_refusal_prompt(method: StructuredOutputMethod) -> PromptTemplate:
    return PromptTemplate.from_template(
        template=refusal_template + get_structured_output_instruction(method),
        template_format="jinja2",
    )
