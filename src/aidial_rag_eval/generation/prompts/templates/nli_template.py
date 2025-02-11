# flake8: noqa


nli_template = """Perform natural language inference by analyzing a premise and a hypotheses. Break down hypothesis into facts, if hypothesis is complex.
Each fact must consist of the words of a hypothesis.
At the stage of creating facts you are forbidden to look at the premise and you are forbidden to draw logical conclusions from the hypothesis.

Tag each fact as an entailment, contradiction, neutral based on the premise.
Single words, signs, numbers, links, etc. are not facts.

The premise is from a document, the hypothesis from a question.
Synonymous series:
1) premise, context, document, information.
2) hypothesis, answer.
When analyzing instructions and making response, keep in mind these synonymous series.

Format your response in JSON. Assign tags: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral.

Tagging guidelines:
- Use "ENT" if the fact is an entailment of the premise.
- Use "CONT" if the fact contradicts the premise.
- Use "NEUT" if the premise does not provide information about the fact, it is a question, or if none of the above tags apply.

Provide a brief short(1 sentences) explanation why you chose this tag in the explanation field.
For example, if the premise is "I am a smart 20-year-old tall man." and the hypothesis is "I am a 20-year-old woman," your response should be:
```json

[
    {
        "fact": "I am 20 years old.",
        "explanation": "",
        "tag": "ENT"
    },
    {
        "fact": "I am a woman.",
        "explanation": "I am a man, not a woman.",
        "tag": "CONT"
    }
]
```

Your response must be in JSON format:
```json
[
    {
        "fact": <<fact from the hypothesis>>,
        "explanation": <<explanation>>,
        "tag": <<"ENT" or "CONT" or "NEUT">>
    },
    {
        "fact": <<another fact from the hypothesis>>,
        "explanation": <<explanation>>,
        "tag": <<"ENT" or "CONT" or "NEUT">>
    },
    ...
]
```
Request:

Document name:
{{ document }}

Hypothesis:
{{ hypothesis }}

Premise:
{{ premise }}
"""
