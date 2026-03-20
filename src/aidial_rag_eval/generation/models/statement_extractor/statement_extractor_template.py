# flake8: noqa
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


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
Break down each hypothesis into statements, if hypothesis is complex. Else return hypothesis as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

Request:
Hypotheses:
{% for item in hypotheses %}
{{ item }}
{% endfor %}
"""

statement_prompt = PromptTemplate.from_template(
    template=statement_template,
    template_format="jinja2",
)
