import json
from typing import Tuple

import numpy as np

from aidial_rag_eval.retrieval.types import (
    ContextChunk,
    ContextRelevance,
    FactMatchResult,
    FactsRanks,
)
from aidial_rag_eval.types import Context, Facts


def highlight_chunk(
    context_chunk: ContextChunk, facts_in_chunk: list[Tuple[int, Tuple[int, int]]]
) -> str:
    chunk_highlight = []

    fact_events = []
    for j, (start, end) in facts_in_chunk:
        fact_events.append((start, False, j))
        fact_events.append((end, True, j))
    fact_events.sort()

    open_facts = set()
    prev_pos = 0
    for pos, is_end, j in fact_events:
        if prev_pos < pos:
            chunk_highlight.append(
                {"text": context_chunk[prev_pos:pos], "facts": list(open_facts)}
            )
        prev_pos = pos

        if is_end:
            open_facts.remove(j)
        else:
            open_facts.add(j)
    if prev_pos < len(context_chunk):
        chunk_highlight.append(
            {"text": context_chunk[prev_pos:], "facts": list(open_facts)}
        )

    return json.dumps({"match": chunk_highlight})


def match_str_facts(
    facts: Facts[str], context: Context, match_fact_func
) -> FactMatchResult:
    facts_ranks: FactsRanks = np.full(len(facts), -1, dtype=int)
    context_relevance: ContextRelevance = np.zeros(len(context), dtype=int)
    context_highlight = []

    for i, c in enumerate(context):
        facts_in_chunk = []
        for j, fact in enumerate(facts):
            match = match_fact_func(fact, c)
            if match is not None:
                if facts_ranks[j] == -1:
                    facts_ranks[j] = i
                else:
                    facts_ranks[j] = min(facts_ranks[j], i)
                facts_in_chunk.append((j, match))
        context_relevance[i] = len(facts_in_chunk)
        context_highlight.append(highlight_chunk(c, facts_in_chunk))

    return FactMatchResult(
        facts_ranks=facts_ranks,
        context_relevance=context_relevance,
        context_highlight=np.array(context_highlight, dtype=object),
    )
