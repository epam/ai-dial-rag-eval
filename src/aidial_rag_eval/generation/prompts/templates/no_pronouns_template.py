# flake8: noqa


no_pronouns_template = """You have the task of replacing all pronouns in a sentences with their corresponding nouns or proper names when their referents are known.
You will be given some sentences.
The first sentence will be given for context only.
Your task is to return all sentences, in which all known pronouns will be replaced by their real names.
It is forbidden to give any explanations in the response.
If the text in the sentence contains no pronouns, doesn't make sense, is a reference, link, a meaningless string of characters, or anything else, then simply return this sentence without any changes.
If you don't know what to do with the sentence, just return original sentence.
You only have permission to change pronouns. You are PROHIBITED from shortening, simplifying sentences, or correcting errors.

For example: "My mom is a good person.", "She always takes care of me." you must return:
```json
{
    "sentences": [
        "My mom is a good person.",
        "My mom always takes care of me.",
        ...
    ]
}
```

Your response template:
```json
{
    "sentences": [
        << first sentence >>,
        << second sentence >>,
        ...
    ]
}
```
Request:
{{ sentences }}"""
