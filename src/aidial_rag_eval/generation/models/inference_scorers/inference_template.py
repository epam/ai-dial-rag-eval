# flake8: noqa
from langchain.prompts import PromptTemplate

inference_template = """
Natural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
A hypothesis is a list of statements provided below. 

{% if document.strip() %}
The name of the document from which the premise was derived is also provided.
{% endif %}

A statement is considered an entailment if it logically follows from the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
Else a statement is considered a neutral.
Important: Don't reject entailment just because of minor extra details - if the main meaning holds, it's still entailment.

For each statement:
Provide a brief short(1 sentences) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.

Format your response in JSON. You must return only JSON.

For example, if the premise is "I am a biology graduate and I work at a tech company." and the list of statements is ["I am a graduate.", "I work at a hospital."] your response should be:

```json
[
    {
        "explanation": "It is true that I am a graduate",
        "tag": "ENT"
    },
    {
        "explanation": "Premise states I work at a tech company, not a hospital.",
        "tag": "CONT"
    }
]
```

Your response must be in JSON format:
```json
[
    {
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
<premise>
{{ premise }}
</premise>

List of statements:
{% for item in statements %}
<statement{{ loop.index }}>
{{ item }}
</statement{{ loop.index }}>
{% endfor %}
"""

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
