import json
from typing import Optional, Tuple

import numpy as np

from rag_eval_metrics.types import (
    Context,
    ContextChunk,
    ContextRelevance,
    FactMatchResult,
    Facts,
    FactsRanks,
)


class CitationMatcher:
    Fact = str

    @staticmethod
    def _canonize(text: str) -> tuple[str, list[int]]:
        text = text.replace("\t", " ")
        text = "".join([c.lower() if c.isalnum() else " " for c in text])

        new_text = ""
        num_skipped_chars_for_pos = []

        # skip duplicated spaces
        for i, c in enumerate(text):
            if c == " " and new_text[-1:] == " ":
                continue
            new_text += c
            num_skipped_chars_for_pos.append(i - len(new_text) + 1)

        num_skipped_chars_for_pos.append(len(text) - len(new_text))

        return new_text, num_skipped_chars_for_pos

    @staticmethod
    def _match(fact: Fact, context_chunk: ContextChunk) -> Optional[Tuple[int, int]]:
        canonized_fact, _ = CitationMatcher._canonize(fact)
        canonized_context_chunk, skipped_chars = CitationMatcher._canonize(
            context_chunk
        )

        start = canonized_context_chunk.find(canonized_fact)
        if start == -1:
            return None
        end = start + len(canonized_fact)

        start = start + skipped_chars[start]
        end = end + skipped_chars[end]
        return start, end

    @staticmethod
    def _highlight_chunk(
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

    @staticmethod
    def match_facts(facts: Facts[Fact], context: Context) -> FactMatchResult:
        facts_ranks: FactsRanks = np.full(len(facts), -1, dtype=int)
        context_relevance: ContextRelevance = np.zeros(len(context), dtype=int)
        context_highlight = []

        for i, c in enumerate(context):
            facts_in_chunk = []
            for j, fact in enumerate(facts):
                match = CitationMatcher._match(fact, c)
                if match is not None:
                    if facts_ranks[j] == -1:
                        facts_ranks[j] = i
                    else:
                        facts_ranks[j] = min(facts_ranks[j], i)
                    facts_in_chunk.append((j, match))
            context_relevance[i] = len(facts_in_chunk)
            context_highlight.append(
                CitationMatcher._highlight_chunk(c, facts_in_chunk)
            )

        return FactMatchResult(
            facts_ranks=facts_ranks,
            context_relevance=context_relevance,
            context_highlight=np.array(context_highlight),
        )
