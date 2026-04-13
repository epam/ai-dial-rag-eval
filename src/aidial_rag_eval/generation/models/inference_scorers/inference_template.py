# flake8: noqa
from langchain_core.prompts import PromptTemplate

inference_template = """
Natural language inference is the task of determining whether the hypothesis is an entailment, contradiction, or neutral with respect to the premise.
A hypothesis is a list of statements provided below.

The name of the document from which the premise was derived is also provided (if available).

A statement is considered an entailment if it is a paraphrase of information expressed in the premise.
A statement is considered a contradiction if it is logically inconsistent with the premise.
A statement is considered neutral if the premise neither supports nor contradicts it, or if the statement contains information the premise does not address.

A statement can be entailed even if the premise contains more information than the statement covers — a statement that restates only part of the premise, or condenses it, is still entailment, as long as it does not introduce new information.

Important: Do not rely on factual world knowledge or logical inference chains to establish entailment — if a fact is not stated in the premise, it is not entailed. However, recognizing synonyms and paraphrases is not inference: it is identifying the same meaning expressed in different words, which is a core part of determining entailment.

For each statement:
Provide a brief short(1 sentence) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.

Format your response in JSON. You must return only JSON.

For example, given the following request:
{
  "document_name": "biology_article",
  "premise": "I am a biology graduate and I work at a tech company.",
  "statements": [
    "I am a graduate.",
    "I work at a hospital.",
    "I am employed at a tech firm."
  ]
}
the expected output is:
{
  "statement_inference": [
    {
      "explanation": "It is true that I am a graduate.",
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

Request:
{{ request_json }}
"""

inference_prompt = PromptTemplate.from_template(
    template=inference_template,
    template_format="jinja2",
)
