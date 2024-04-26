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
    def _canonize(text: str) -> str:
        text = text.replace("\t", "")
        text = "".join([c.lower() for c in text if c.isalnum() or c.isspace()])
        text = " ".join(text.split())
        return text

    @staticmethod
    def _match(fact: Fact, context_chunk: ContextChunk) -> bool:
        canonized_fact = CitationMatcher._canonize(fact)
        canonized_context_chunk = CitationMatcher._canonize(context_chunk)
        return canonized_fact in canonized_context_chunk

    @staticmethod
    def match_facts(facts: Facts[Fact], context: Context) -> FactMatchResult:
        facts_ranks: FactsRanks = np.full(len(facts), -1, dtype=int)
        context_relevance: ContextRelevance = np.zeros(len(context), dtype=int)

        for i, fact in enumerate(facts):
            chunk_index = next(
                (j for j, c in enumerate(context) if CitationMatcher._match(fact, c)),
                -1,
            )
            facts_ranks[i] = chunk_index
            if chunk_index >= 0:
                context_relevance[facts_ranks[i]] += 1

        return FactMatchResult(
            facts_ranks=facts_ranks, context_relevance=context_relevance
        )
