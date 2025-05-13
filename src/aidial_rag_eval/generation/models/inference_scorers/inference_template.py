# flake8: noqa
from langchain.prompts import PromptTemplate

inference_template = """
Natural language inference is the task of determining the relationship between a premise and a hypothesis, specifically whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
{% if document.strip() %}
The name of the document from which the premise was derived is also provided.
{% endif %}
First: break down hypothesis into statements, if hypothesis is complex.
At the stage of creating statements you are forbidden to look at the premise and you are forbidden to draw logical conclusions from the hypothesis.
A statement is a declarative self-contained substring derived from the hypothesis.


Second: Tag each statement as an entailment, contradiction, neutral based on the premise. At this stage you are allowed to draw logical conclusions from the hypothesis.

A statement is considered an entailment if it logically follows from the premise.
A statement is identified as a contradiction if it is logically inconsistent with the premise.
Else a statement is labeled as neutral, or if the statement is a question or unrelated to the premise.
Single words, signs, numbers, links, etc. are not statements.

Provide a brief short(1 sentences) explanation of whether the statement is an entailment, contradaction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.
Format your response in JSON. You must return only JSON.

For example, if the premise is "I am a smart 20-year-old tall man." and the hypothesis is "I am a 20-year-old woman," your response should be:
```json

[
    {
        "statement": "I am 20 years old.",
        "explanation": "",
        "tag": "ENT"
    },
    {
        "statement": "I am a woman.",
        "explanation": "I am a man, not a woman.",
        "tag": "CONT"
    }
]
```

Your response must be in JSON format:
```json
[
    {
        "statement": <<statement from the hypothesis>>,
        "explanation": <<explanation>>,
        "tag": <<"ENT" or "CONT" or "NEUT">>
    },
    {
        "statement": <<another statement from the hypothesis>>,
        "explanation": <<explanation>>,
        "tag": <<"ENT" or "CONT" or "NEUT">>
    },
    ...
]
```
Request:

{% if document.strip() %}
<document_name>
{{ document }}
</document_name>
{% endif %}
<hypothesis>
{{ hypothesis }}
</hypothesis>

<premise>
{{ premise }}
</premise>
"""

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
