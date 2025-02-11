from langchain.prompts import PromptTemplate

from aidial_rag_eval.generation.prompts.templates.nli_template import nli_template
from aidial_rag_eval.generation.prompts.templates.no_pronouns_template import (
    no_pronouns_template,
)
from aidial_rag_eval.generation.prompts.templates.refusal_template import (
    refusal_template,
)

nli_prompt = PromptTemplate.from_template(
    template=nli_template,
    template_format="jinja2",
)

no_pronouns_prompt = PromptTemplate.from_template(
    template=no_pronouns_template,
    template_format="jinja2",
)

refusal_prompt = PromptTemplate.from_template(
    template=refusal_template,
    template_format="jinja2",
)


__all__ = [
    "nli_prompt",
    "no_pronouns_prompt",
    "refusal_prompt",
]
