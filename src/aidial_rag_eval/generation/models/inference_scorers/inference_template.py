# flake8: noqa
from langchain.prompts import PromptTemplate

neutral_contradiction_template = """
### Task
You are given: a premise and a list of statements.
The statements in the list are derived from a same sentence, i.e. they are related to each other.

For each statement:
Explain why the statement does not entail the premise.

If the statement does not entail the premise:
Quote the part of the premise that does not support the statement and briefly explain why, or explain why the statement is unrelated to the premise.
The reasons why a statement cannot be considered an entailment can be of two types: neutral and contradiction.
If the statement contains more than one issue, mention all the problems you can identify.

If you cannot identify a lack of entailment, write: "Failed to find a lack of entailment."

### Output
Format your response in JSON. You must return only JSON.
Reasoning should be provided as plain text for each statement.
The output JSON must represent a simple list of strings, without any nesting.

Your response must be in JSON format:
```json
[
    <<reasoning for the 1st statement>>,
    <<reasoning for the 2ond statement>>,
    ...
]
```
### Example
For example, if the premise is "I am a biology graduate and I work at a tech company." and the list of statements is ["I am a graduate.", "I work at a hospital.", "I am bored."] your response should be:

```json
[
    "Failed to find a lack of entailment.",
    "Premise states 'I work at a tech company', not a hospital. The statement purely contradicts the premise.",
    "The premise does not mention the person's emotional state, that is why I think the statement is neutral."
]
```

### Request

{% if document.strip() %}
The name of the document from which the premise was derived is provided.
{% endif %}

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

neutral_contradiction_prompt = PromptTemplate.from_template(
    template=neutral_contradiction_template,
    template_format="jinja2",
)

entailment_template = """
### Task
You are given: a premise and a list of statements.
The statements in the list are derived from a same sentence, i.e. they are related to each other.

For each statement:
Provide reasoning on whether statement is an entailment with respect to the premise or not.

If you believe that a statement is entailed by the premise, you must:
Quote parts of the premise in a logical sequence from which the statement follows, and for each
quote you must provide an explanation of why it helps to prove that the statement is entailed by the premise.
If you cannot prove the entailment, you must write: "Failed to entail the statement from the premise."

### Output
Format your response in JSON. You must return only JSON.
Reasoning should be provided as plain text for each statement.
The output JSON must represent a simple list of strings, without any nesting.

Your response must be in JSON format:
```json
[
    <<reasoning for the 1st statement>>,
    <<reasoning for the 2ond statement>>,
    ...
]
```

### Example
For example, if the premise is "I am a biology graduate and I work at a tech company." and the list of statements is ["I am a graduate.", "I work at a hospital."] your response should be:

```json
[
    "The premise states 'I am a biology graduate,' which means the statement is entailed by the premise because the premise explicitly mentions a degree, even specifying a 'biology' degree. Therefore, the statement clearly follows."
    "Failed to entail the statement from the premise."
]
```

### Request

{% if document.strip() %}
The name of the document from which the premise was derived is provided.
{% endif %}

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

entailment_prompt = PromptTemplate.from_template(
    template=entailment_template,
    template_format="jinja2",
)


inference_template = """
### Task
Natural language inference is the task of determining whether the statement is an entailment, contradiction, or neutral with respect to the premise.
A list of statements provided below.
The statements in the list are derived from a same sentence, i.e. they are related to each other.

A statement is considered an entailment if it logically follows from the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
Else a statement is considered a neutral.

For each statement:
Provide an reasoning of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your reasoning: "Entailment" for entailment, "Contradiction" for contradiction, "Neutral" for neutral or if none of the above tags apply.

You are being presented with arguments promoting a specific conclusion(counterarguments for neutral or contradiction tags and arguments for entailment tags).

### Reasoning:
The reasoning is constructed in a step-by-step manner:
1. Comment on the arguments about the relationship between the statement and the premise, if they are provided.
2. Provide your own comments on the relationship between the statement and the premise.
3. Draw a final conclusion.

Construct your reasoning in this order and never start with the conclusion on the first step - just assess the validity of the arguments. 

### Output
Format your response in JSON. You must return only JSON.

Your response must be in JSON format:
```json
[
    {
        "reasoning": <<reasoning>>,
        "tag": <<"Entailment" or "Contradiction" or "Neutral">>
    },
    ...
]
```

### Example
For example, if the premise is "I am a biology graduate and I work at a tech company.",
statements: ["I am a graduate.", "I work at a hospital.", "I am bored."].
counterarguments: ["The statement is baseless.", "Premise states "I work at a tech company", not a hospital.", "The statement is baseless.'"]
arguments: ["Premise explicitly states that 'I am a biology graduate.' means statement clearly follows.", "Premise explicitly states that 'I work' means statement clearly follows.", "Failed to entail the statement from the premise."]
```json
[
    {
        "reasoning": "The counterargument is not valid because the statement is based on the premise. The argument is valid because the premise states 'I am a biology graduate,' which means the statement is directly quoted from the premise. Part of the premise essentially states the same thing as the statement, just with more detail, which means the less specific statement can be inferred from it. The statement is an entailment.",
        "tag": "Entailment"
    },
    {
        "reasoning": "The counterargument is valid because the premise states 'I work at a tech company,' not a hospital. The argument is not valid because 'I work' does not entail 'I work at a hospital.' The workplace mentioned in the statement clearly contradicts the one in the premise. The statement is a contradiction.",
        "tag": "Contradiction"
    },
    {
        "reasoning": "The counterargument is valid because the premise does not mention the person's emotional state, making the statement baseless. The premise does not mention the person's emotional state. The statement is neutral.",
        "tag": "Neutral"
    }
]
```

### Request

{% if document.strip() %}
The name of the document from which the premise was derived is provided.
{% endif %}

{% if document.strip() %}
<document_name>
{{ document }}
</document_name>
{% endif %}
<premise>
{{ premise }}
</premise>

List of statements with arguments:
{% for item in statements %}
<statement{{ loop.index }}>
{{ item }}
</statement{{ loop.index }}>
<argument{{ loop.index }}>
{{ entailment_arguments[loop.index] }}
</argument{{ loop.index }}>
<counterargument{{ loop.index }}>
{{ neutral_contradiction_arguments[loop.index] }}
</counterargument{{ loop.index }}>
{% endfor %}
"""

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
