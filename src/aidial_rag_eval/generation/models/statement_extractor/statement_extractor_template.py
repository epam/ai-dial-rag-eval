# flake8: noqa
from langchain_core.prompts import PromptTemplate

statement_template = """
Break down each hypothesis into statements, if hypothesis is complex. Else if the hypothesis is already a single statement, return it unchanged as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

When a hypothesis contains an enumeration or list, split it so that each item in the list becomes a separate statement. Preserve the relationship from the parent clause in each statement.

Examples:
Hypothesis 1: "The sky is blue and the grass is green."
Hypothesis 2: "Water boils at 100 degrees Celsius."
Hypothesis 3: "The company has offices in Paris, London, and Berlin."

Expected output:
- hypothesis1 → ["The sky is blue.", "The grass is green."]
- hypothesis2 → ["Water boils at 100 degrees Celsius."]
- hypothesis3 → ["The company has an office in Paris.", "The company has an office in London.", "The company has an office in Berlin."]

Your response must be in JSON format:
```json
{
    "hypothesis_statements": [
        {
            "statements": [
                <<statement1 from the first hypothesis>>,
                <<statement2 from the first hypothesis>>,
                ...
            ]
        },
        {
            "statements": [
                <<statement1 from the second hypothesis>>,
                <<statement2 from the second hypothesis>>,
                ...
            ]
        },
        ...
    ]
}
```

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
