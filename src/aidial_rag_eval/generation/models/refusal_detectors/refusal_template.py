# flake8: noqa
from langchain_core.prompts import PromptTemplate

refusal_template = """
Answer Refusal task is to determine if an answer should be tagged as a refusal to answer based on specific criteria.
The criteria include identifying if the answer explicitly states something is wrong with the question, request, or premise, or if it indicates a lack of information, irrelevance, or refusal to answer.

Single words, signs, numbers, links, etc. are not considered refusal to answer.

Lack of information can be formulated in different ways, pay attention to the list of synonyms.
Synonymous series:
1) premise, context, document, information.
2) hypothesis, answer.

Tagging guidelines:
- Use "REJ" if the answer is answer refusal, else tag it "ANS".

For example, given the following list of answers:
["There is no answer to this question.", "The answer to this question is yes."]
the expected output is:
{"tags": ["REJ", "ANS"]}

Each answer from the list of answers corresponds to a tag in your response.
The first answer corresponds to the first tag, the second corresponds to the second.
The number of tags must be the same as the number of answers in the answer list.
Each answer, even meaningless, must have it's own tag, if you don't know how to tag it, than just leave "ANS" tag for it.

Request:
List of answers:
{{ answers_json }}"""

refusal_prompt = PromptTemplate.from_template(
    template=refusal_template,
    template_format="jinja2",
)
