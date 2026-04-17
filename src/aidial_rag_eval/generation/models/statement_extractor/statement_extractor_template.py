# flake8: noqa
from langchain_core.prompts import PromptTemplate

statement_template = """
Break down each hypothesis into statements, if hypothesis is complex. Else if the hypothesis is already a single statement, return it unchanged as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

When a hypothesis contains an enumeration or list, split it so that each item in the list becomes a separate statement. Preserve the relationship from the parent clause in each statement.

The output list must contain exactly as many items as there are input hypotheses, in the same order: the first item corresponds to the first hypothesis, the second item to the second hypothesis, and so on.

Examples:
Input hypotheses (JSON array of strings, one hypothesis per element):
[
  "The sky is blue and the grass is green.",
  "Water boils at 100 degrees Celsius.",
  "The company has offices in Paris, London, and Berlin."
]

Expected output:
{
  "hypothesis_statements": [
    {
      "statements": [
        "The sky is blue.",
        "The grass is green."
      ]
    },
    {
      "statements": [
        "Water boils at 100 degrees Celsius."
      ]
    },
    {
      "statements": [
        "The company has an office in Paris.",
        "The company has an office in London.",
        "The company has an office in Berlin."
      ]
    }
  ]
}

Request:
Input hypotheses (JSON array of strings, one hypothesis per element):
{{ hypotheses_json }}
"""

statement_prompt = PromptTemplate.from_template(
    template=statement_template,
    template_format="jinja2",
)
