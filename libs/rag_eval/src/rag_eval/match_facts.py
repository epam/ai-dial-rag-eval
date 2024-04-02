import numpy as np


def canonize(text):
    text = text.replace("\t", "")
    # delete all non alphanumeric characters and make lowercase
    text = "".join([c.lower() for c in text if c.isalnum() or c.isspace()])
    # replace multiple spaces with one
    text = " ".join(text.split())
    return text


def match_facts(facts: list[str], context: list[str]) -> tuple[np.ndarray, np.ndarray]:
    facts_ranks = np.full(len(facts), -1, dtype=int)
    context_relevance = np.zeros(len(context), dtype=int)

    for i, fact in enumerate(facts):
        chunk_index = next(
            (j for j, c in enumerate(context) if canonize(fact) in canonize(c)), -1
        )
        facts_ranks[i] = chunk_index
        if chunk_index >= 0:
            context_relevance[facts_ranks[i]] += 1

    return facts_ranks, context_relevance
