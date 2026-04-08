# flake8: noqa
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
    get_structured_output_instruction,
)


class HypothesisStatements(BaseModel):
    """Statements extracted from a single hypothesis."""

    statements: List[str] = Field(
        description="Declarative statements extracted from one hypothesis."
    )


class StatementsOutput(BaseModel):
    """The list must contain exactly as many items as there are input hypotheses. The first item corresponds to hypothesis1, the second item to hypothesis2, and so on."""

    hypothesis_statements: List[HypothesisStatements] = Field(
        description=(
            "One item per input hypothesis, in the same order. "
            "The first item contains statements for hypothesis1, "
            "the second item contains statements for hypothesis2, etc."
        )
    )


statement_template = """
Break down each hypothesis into statements, if hypothesis is complex. Else if the hypothesis is already a single statement, return it unchanged as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

When a hypothesis contains an enumeration or list, split it so that each item in the list becomes a separate statement. Preserve the relationship from the parent clause in each statement.

Examples:
Input hypotheses: ["The sky is blue and the grass is green.", "Water boils at 100 degrees Celsius.", "The company has offices in Paris, London, and Berlin."]

Expected output:
- hypothesis1 → statements: ["The sky is blue.", "The grass is green."]
- hypothesis2 → statements: ["Water boils at 100 degrees Celsius."]
- hypothesis3 → statements: ["The company has an office in Paris.", "The company has an office in London.", "The company has an office in Berlin."]

Request:
Input hypotheses:
{{ hypotheses_json }}
"""


def get_statement_prompt(method: StructuredOutputMethod) -> PromptTemplate:
    return PromptTemplate.from_template(
        template=statement_template + get_structured_output_instruction(method),
        template_format="jinja2",
    )
