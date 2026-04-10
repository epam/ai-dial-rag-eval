# flake8: noqa
from typing import List, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from aidial_rag_eval.generation.models.structured_output_utils import (
    StructuredOutputMethod,
    get_example_output_note,
    get_structured_output_instruction,
)


class StatementInference(BaseModel):
    """Inference result for a single statement."""

    explanation: str = Field(description="Brief explanation of the inference decision.")
    tag: Literal["ENT", "CONT", "NEUT"] = Field(
        description='"ENT" if the statement is entailed by the premise, "CONT" if it contradicts the premise, "NEUT" otherwise.'
    )


class StatementInferenceOutput(BaseModel):
    """The list must contain exactly as many items as there are input statements. The first item corresponds to statement1, the second to statement2, and so on."""

    statement_inference: List[StatementInference] = Field(
        description=(
            "One item per input statement, in the same order. "
            "The first item corresponds to statement1, the second to statement2, etc."
        )
    )


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
Provide a brief short(1 sentences) explanation of whether the statement is an entailment, contradiction or neutral with respect to the premise.
Assign tags based on your explanation: "ENT" for entailment, "CONT" for contradiction, "NEUT" for neutral or if none of the above tags apply.

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


def get_inference_prompt(method: StructuredOutputMethod) -> PromptTemplate:
    return PromptTemplate.from_template(
        template=inference_template
        + get_example_output_note(method)
        + get_structured_output_instruction(method),
        template_format="jinja2",
    )
