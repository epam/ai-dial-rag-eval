# flake8: noqa


refusal_template = """Evaluate each hypothesis and determine if it should be tagged as a refusal to answer based on the criteria provided.

Single words, signs, numbers, links, etc. are not considered refusal to answer.

Synonymous series:
1) premise, context, document, information.
2) hypothesis, answer.

Tagging guidelines:
- Use "REJ" if the hypothesis explicitly states something is wrong with the question/request/premise.
- Lack of information can be formulated in different ways, pay attention to the list of synonyms.
- Use "REJ" in these scenarios (by information we mean any word from its synonymous series):
  1. The hypothesis is an explicit question or an explicit request for information or to change topic.
  2. The hypothesis explicitly states there's missing or insufficient information in the premise.
  3. The hypothesis explicitly states it's out of scope or irrelevant to the premise.
  4. The hypothesis is an explicit refusal to answer the question.
  5. Else tag it "ANS".


Example:
An explicit statement: "There is no answer to this question." should be tagged "REJ".
A statement that is not explicit: "The answer to this question is yes." should be tagged "ANS".

Format your response in JSON:
```json
[
    <<"REJ" or "ANS">>,
    <<"REJ" or "ANS">>,
    ...
]
```

Each hypothesis from the list of hypotheses corresponds to a tag in your response.
The first hypothesis corresponds to the first tag, the second corresponds to the second.
The number of tags must be the same as the number of hypotheses in the hypothesis list.
Each hypothesis, even meaningless, must have it's own tag, if you don't know how to tag it, than just leave "ANS" tag for it.

Request:
List of hypotheses:
{% for item in answers %}
Hypothesis {{ loop.index }}:
{{ item }}
{% endfor %}"""
