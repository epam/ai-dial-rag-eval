from typing import Optional, Tuple

from aidial_rag_eval.facts.match_str_facts import match_str_facts
from aidial_rag_eval.retrieval.types import ContextChunk, FactMatchResult
from aidial_rag_eval.types import Context, Facts


class ExactStringMatcher:
    Fact = str

    @staticmethod
    def _match(fact: Fact, context_chunk: ContextChunk) -> Optional[Tuple[int, int]]:
        if fact == context_chunk:
            return (0, len(context_chunk))
        return None

    @staticmethod
    def match_facts(facts: Facts[Fact], context: Context) -> FactMatchResult:
        return match_str_facts(facts, context, ExactStringMatcher._match)
