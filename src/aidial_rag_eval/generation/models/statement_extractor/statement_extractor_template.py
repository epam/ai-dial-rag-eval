# flake8: noqa
from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


class HypothesisSegmentStatementsOutput(BaseModel):
    statements: List[str]


class HypothesisStatementsOutput(BaseModel):
    hypothesis_statements: List[HypothesisSegmentStatementsOutput]


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
