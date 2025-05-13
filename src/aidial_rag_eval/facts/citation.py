from typing import Optional, Tuple

from aidial_rag_eval.facts.match_str_facts import match_str_facts
from aidial_rag_eval.retrieval.types import ContextChunk, FactMatchResult
from aidial_rag_eval.types import Context, Facts


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
    def match_facts(facts: Facts[Fact], context: Context) -> FactMatchResult:
        return match_str_facts(facts, context, CitationMatcher._match)
