# flake8: noqa
from langchain.prompts import PromptTemplate

statement_template = """
Break down each hypothesis into statements, if hypothesis is complex. Else return hypothesis as a single statement.

A statement is a declarative independent self-contained non-overlapping substring forming a complete sentence derived from the hypothesis.

Single words, signs, numbers, links, etc. are not statements.

Your response must be in JSON format:
```json
[
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
```

Request:
Hypotheses:
{% for item in hypotheses %}
<hypothesis{{ loop.index }}>
{{ item }}
</hypothesis{{ loop.index }}>
{% endfor %}
"""

statement_prompt = PromptTemplate.from_template(
    template=statement_template,
    template_format="jinja2",
)
