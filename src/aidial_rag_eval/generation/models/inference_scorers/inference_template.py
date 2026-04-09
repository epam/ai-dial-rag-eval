# flake8: noqa
from langchain_core.prompts import PromptTemplate

inference_template = """
Natural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
A hypothesis is a list of statements provided below. 

{% if document %}
The name of the document from which the premise was derived is also provided.
{% endif %}

A statement is considered an entailment if it is a paraphrase of information expressed in the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
A statement is considered neutral if the premise neither supports nor contradicts it, or if the statement contains information the premise does not address.

A statement can be entailed even if the premise contains additional details not mentioned in the statement — a subset or summary of the premise is still entailment.
However, if the statement introduces information not expressed in the premise, it is not entailment.

Important: Base your decision on whether the premise expresses the same information, not on what can be inferred from it.
Do not use general knowledge, logical inference, or draw conclusions beyond what the premise expresses.
If the premise is silent on some aspect of the statement, treat that aspect as not supported.

For each statement:
Provide a brief short(1 sentences) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.

Format your response in JSON. You must return only JSON.

For example, if the premise is "I am a biology graduate and I work at a tech company." and the list of statements is ["I am a graduate.", "I work at a hospital.", "I am employed at a tech firm."] your response should be:

```json
{
    "statement_inference": [
        {
            "explanation": "It is true that I am a graduate",
            "tag": "ENT"
        },
        {
            "explanation": "Premise states I work at a tech company, not a hospital.",
            "tag": "CONT"
        },
        {
            "explanation": "Employed at a tech firm is a paraphrase of working at a tech company.",
            "tag": "ENT"
        }
    ]
}
```

Your response must be in JSON format:
```json
{
    "statement_inference": [
        {
            "explanation": <<explanation>>,
            "tag": <<"ENT" or "CONT" or "NEUT">>
        },
        ...
    ]
}
```
Request:

{% if document %}
<document_name>
{{ document }}
</document_name>
{% endif %}
<premise>
{{ premise }}
</premise>

List of statements:
{% for item in statements %}
{{ item }}
{% endfor %}
"""

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
