import pandas as pd
import json
from openai import AzureOpenAI


def to_context(doc_chunks):
    return [attachment["data"] for attachment in doc_chunks]

def find_stage(stages, stage_name):
    for stage in stages:
        if stage["name"].startswith(stage_name):
            return stage
    return None


def ask_dial_app(question, app_name, retrieval_stage=None, messages_template='[{{ "role": "user", "content": "{}" }}]'):
    dial_client = AzureOpenAI(
        azure_endpoint = 'https://dev-dial-core.staging.deltixhub.io',
        api_version="2023-05-15"
    )

    messages = json.loads(messages_template.format(question))
    res = dial_client.chat.completions.create(
        model=app_name,
        messages=messages,
    )

    ans_message = res.choices[0].message
    answer = ans_message.content
    doc_chunks = []
    try:
        if retrieval_stage:
            stage = find_stage(ans_message.custom_content["stages"], retrieval_stage)
            doc_chunks = stage["attachments"]
        else:
            doc_chunks = ans_message.custom_content["attachments"]
    except Exception as e:
        print(repr(e))
        
    return pd.Series({
        "question": question,
        "answer": answer,
        "context": to_context(doc_chunks)
    })