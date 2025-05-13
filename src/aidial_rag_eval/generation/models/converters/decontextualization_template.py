# flake8: noqa
from langchain.prompts import PromptTemplate

decontextualization_template = """
The task is to replace all pronouns in a segments with their corresponding nouns or proper names when their referents are known.
You will receive segments.
If a segment is nonsensical, a reference, link, or meaningless, return it unchanged.
If unsure what to do with segment, return the original segment.
Only perform the task; do not shorten, simplify, or correct errors.
Do not provide explanations.

For example: "My mom is a good person.", "She always takes care of me." you must return:
```json
{
    "segments": [
        "My mom is a good person.",
        "My mom always takes care of me.",
        ...
    ]
}
```

Your response template:
```json
{
    "segments": [
        << Segment 1 >>,
        << Segment 2 >>,
        ...
    ]
}
```
List of input segments:
```json
{
    "segments": [{% for item in sentences %}
        "{{ item }}"{% if not loop.last %},{% endif %}{% endfor %}
    ]
}
```
Important: before generating response, check the number and structure of segments. The response must have the same number of segments, split the same way.
"""

decontextualization_prompt = PromptTemplate.from_template(
    template=decontextualization_template,
    template_format="jinja2",
)
