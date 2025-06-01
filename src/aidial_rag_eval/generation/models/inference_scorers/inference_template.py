# flake8: noqa
from langchain.prompts import PromptTemplate

inference_template = """
### Task
Natural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
A hypothesis is a list of statements provided below. 

A statement is considered an entailment if it logically follows from the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
Else a statement is considered a neutral.

For each statement:
Provide an reasoning of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your reasoning: "Entailment" for entailment, "Contradiction" for contradiction, "Neutral" for neutral or if none of the above tags apply.

### Reasoning:
You need to create an reasoning that answers the question of how the statement relates to the premise.
Reasoning construction algorithm:

First, generate 3 hypothesis based on the premise:
1. Contradiction hypothesis:
Start the sentence with "Contradiction hypothesis:" and IDENTIFY a fact in the premise that contradicts the statement.
If this is not possible, write "no such hypothesis".
2. Neutral hypothesis:
Star with "Neutral hypothesis:" and CREATE a fact that is consistent with the premise but contradicts the statement.
If not possible, write "no such hypothesis".
3. Entailment hypothesis:
Start with "Entailment hypothesis:" and DERIVE a fact from the premise that entails the statement.
If not possible, write "no such hypothesis".

Next, you should write one sentence discussing how valid the hypotheses are.
By "is valid" we mean re-evaluating each hypothesis against the premise and statement.
Remember: the hypotheses from the first step are often wrong, so skepticism is healthy. Avoid abstract reasoning -
don't claim a hypothesis is true without quoting specific sources.

Then, make a final judgment by proceeding in order:
a) If the Contradiction hypothesis is valid, write:
"The statement is a contradiction."
b) Otherwise, if the Neutral hypothesis is valid, write:
"The statement is a neutral."
c) Otherwise, if the Entailment hypothesis is valid, write:
"The statement is a entailment."
d) If none are valid, write:
"The statement is a neutral."

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
For example, if the premise is "I am a biology graduate and I work at a tech company." and the list of statements is ["I am a graduate.", "I work at a hospital.", "I am bored."] your response should be:

```json
[
    {
        "reasoning": "Contradiction hypothesis: no such hypothesis. Neutral hypothesis: "My friend is a biology graduate." Entailment hypothesis: I am a biology graduate. Premise states "I am a biology graduate" means Entailment hypothesis is valid because the hypothesis is a direct quote from the preimse and the statement clearly follows from the hypothesis, while Neutral hypothesis is not valid because both me and my friend both can be biology graduate so Neutral hypothesis doesn't contradict the statement. The statement is a entailment.",
        "tag": "Entailment"
    },
    {
        "reasoning": "Contradiction hypothesis: I work at a tech company. Neutral hypothesis: I work at a sales company. Entailment hypothesis: I work. Premise states "I work at a tech company", not a hospital, that is why Contradiction hypothesis is valid, Neutral hypothesis isn't valid because "I work at a sales company." contradicts the premise and Entailment hypothesis is not valid because "I work" doesn't entail "I work at a hospital.". The statement is a contradiction.",
        "tag": "Contradiction"
    },
    {
        "reasoning": "Contradiction hypothesis: I am having fun. Neutral hypothesis: I am having fun. Entailment hypothesis: no such hypothesis. The premise does not mention the person's emotional state, that is why I think Neutral hypothesis is valid and Contradiction hypothesis is not. The statement is a neutral.",
        "tag": "Neutral"
    }
]
```

Note that the example includes flawed hypotheses meant to be viewed critically.
When answering a request feel free to write "no such hypothesis".

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

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
