import json
from json import JSONDecodeError

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI


def parse_facts(facts_raw):
    json_str = facts_raw
    if "```json" in facts_raw:
        json_str = facts_raw.split("```json")[1].split("```")[0].strip()

    try:
        # return json.loads(json_str)["facts"]
        return json.loads(json_str, strict=False)["citations"]
    except JSONDecodeError as e:
        print(facts_raw)
        print(json_str)
        print(e)
        print()
        return None


def from_context(context):
    if isinstance(context, list):
        return context
    if isinstance(context, np.ndarray):
        return context.tolist()

    if "<document>" not in context:
        return [context]
    context = context.replace("<context>\n", "").replace("\n</context>", "")
    doc_chunks = []
    for doc in context.split("<document>")[1:]:
        doc = doc.replace("\n</document>", "")
        doc_chunks.append(doc)
    return doc_chunks


def canonize(text):
    text = text.replace("\t", "")
    # delete all non alphanumeric characters and make lowercase
    text = "".join([c.lower() for c in text if c.isalnum() or c.isspace()])
    # replace multiple spaces with one
    text = " ".join(text.split())
    return text


def validate_facts(row, debug=True):
    facts = row["facts"]
    if not facts:
        if debug:
            print("No facts")
        return False
    # context = json.dumps(from_context(row["context"]), indent=2)
    # missing = [fact for fact in facts if fact not in context]
    context = "\n".join(from_context(row["context"]))
    context = canonize(context)
    facts = [canonize(fact) for fact in facts]
    missing = [fact for fact in facts if fact not in context]
    if debug and missing:
        print(f"Missing facts:\n{[m.encode('utf-8') for m in missing]}")
        print(f"Context:\n {context.encode('utf-8')}")
        print()
        print()
        print()

    return not missing


def classify_validation_result(row):
    if row["facts_is_valid"]:
        return "valid"
    if row["facts"] is None:
        return "bad_json"
    if row["facts"] == []:
        return "empty_list"
    return "invalid"


EXTRACT_FACTS_PROMPT = ChatPromptTemplate.from_template(
    """
You are an citation extraction system. Your task is to extract the citations from the source documents which were used in the answer.

You are given the following text from the source documents:

```json
{context}
```

Some other question answering system has provided the following passage:

```
Question: {question}
Answer: {text}
```

Please, extract the citations from the source documents which were used to generate this answer and can be used to verify that the answer is correct.
The sentences should be the cited exactly as in the source documents, because it will be automatically matched with the text of the source documents as a strings.
Cite the whole sentence from the beginning to the end.
Write the sentences as a JSON with a list of strings with the key "citations". The JSON should be valid, as it will be automatically parsed.
The JSON should be wrapped in a markdown code block with the language "json". Do not write anything else in the response, as it will be automatically parsed.
If there are no citations to support the correctness of the answer, return an empty list.

Example of the response format:
```json
{{
    "citations": [
        "The first citation from the source documents",
        "The second citation from the source documents"
    ]
}}
```
"""  # noqa: B950
)


def extract_facts_raw_mixtral_iteration(row, temperature=0, answer_column="answer"):
    model = AzureChatOpenAI(
        # deployment_name='Mixtral-8x7B-Instruct-v0.1',
        deployment_name="gpt-4-32k",
        azure_endpoint="https://dev-dial-core.staging.deltixhub.io",
        openai_api_version="2023-03-15-preview",
        verbose=True,
        streaming=True,
        temperature=temperature,
        max_tokens=1000,
    )

    chain = EXTRACT_FACTS_PROMPT | model | StrOutputParser()

    result = chain.invoke(
        {
            "question": row["question"],
            # "context": row["context"],
            "context": json.dumps(from_context(row["context"]), indent=2),
            "text": row[answer_column],
        },
    )
    return result


def extract_facts_raw_mixtral(row, answer_column="answer"):
    # TODO : Configure range of temperatures
    for i in range(1):
        temperature = max(i, 1)
        result = extract_facts_raw_mixtral_iteration(row, temperature, answer_column)
        facts = parse_facts(result)
        facts_is_valid = validate_facts(
            {"facts": facts, "context": row["context"]}, False
        )
        validation_result = classify_validation_result(
            {"facts": facts, "facts_is_valid": facts_is_valid}
        )
        if validation_result != "bad_json":
            pd.Series({"facts": facts, "validation_result": validation_result})
            return facts, validation_result
    return pd.Series({"facts": None, "validation_result": "bad_json"})
