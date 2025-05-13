from typing import Callable, List, Optional, Tuple

import nltk
from more_itertools import chunked

from aidial_rag_eval.generation.types import Text, TextSegment

Splitter = Callable[[TextSegment], List[TextSegment]]
SegmentChecker = Callable[[TextSegment], bool]


def get_delimiters(text: Text, segmented_text: List[TextSegment]) -> List[str]:
    delimiters = []
    start = 0
    for i, part in enumerate(segmented_text[:-1]):
        start += len(part)
        if start < len(text):
            delimiter = ""
            while start < len(text) and not text[start:].startswith(
                segmented_text[i + 1]
            ):
                delimiter += text[start]
                start += 1
            delimiters.append(delimiter)
    return delimiters


def join_with_delimiters(
    segmented_text: List[TextSegment], delimiters: List[str]
) -> Text:
    result = []
    for i in range(len(segmented_text)):
        result.append(segmented_text[i])
        if i < len(delimiters):
            result.append(delimiters[i])
    return "".join(result)


def _get_split_with_delimiters(
    original_segment: TextSegment, splitter: Splitter
) -> Tuple[List[TextSegment], List[str]]:
    segments = splitter(original_segment)
    if len(segments) > 1:
        delimiters = get_delimiters(original_segment, segments)
    else:
        delimiters = []
    assert len(segments) - 1 == len(delimiters)
    return segments, delimiters


def _default_checker(original_segment: TextSegment) -> bool:
    return True


class SegmentedText:
    """
    The SegmentedText class is designed to store
    the original text in a format that is divided into segments.

    Along with the segments, delimiters are also stored.
    When the segments are joined with the delimiters,
    the original text can be reconstructed, if the segments have not been replaced.
    """

    segments: List[TextSegment]
    """A list of segments into which the original text has been divided."""

    delimiters: List[str]
    """The gaps between segments.
    When segments are joined using these delimiters, they reconstruct the original text.
    len(delimiters) == len(segments) - 1."""

    def __init__(
        self,
        segments: List[TextSegment],
        delimiters: List[str],
    ):
        assert len(segments) - 1 == len(delimiters)
        self.segments = segments.copy()
        self.delimiters = delimiters.copy()

    @classmethod
    def from_text(cls, text: Text) -> "SegmentedText":
        nltk.download("punkt_tab", quiet=True)

        max_len = 500
        min_len = 10
        conditional_splitters: List[Splitter] = [
            lambda x: x.split("\n\n"),
            lambda x: x.split("\n"),
            lambda x: list(map(lambda y: "".join(y), chunked(x, max_len))),
        ]
        segmented_text = cls([text], [])
        segmented_text = apply_splitter_to_segmented_text(
            segmented_text, nltk.sent_tokenize
        )
        for splitter in conditional_splitters:
            segmented_text = apply_splitter_to_segmented_text(
                segmented_text, splitter, lambda x: len(x) > max_len
            )

        segmented_text = join_segments_to_next_by_condition(
            segmented_text, lambda x: len(x) < min_len
        )
        return segmented_text

    def replace_segments(self, segments: List[TextSegment], start_idx: int):
        self.segments[start_idx : start_idx + len(segments)] = segments

    def get_joined_segments_by_range(self, start: int, end: int) -> str:
        return join_with_delimiters(
            self.segments[start:end], self.delimiters[start : end - 1]
        )


def apply_splitter_to_segmented_text(
    segmented_text: SegmentedText,
    splitter: Splitter,
    checker: Optional[SegmentChecker] = None,
) -> SegmentedText:
    if checker is None:
        checker = _default_checker
    new_segments = []
    new_delimiters = []

    for i, original_segment in enumerate(segmented_text.segments):
        split_segment, split_delimiters = [original_segment], []
        if checker(original_segment):
            split_segment, split_delimiters = _get_split_with_delimiters(
                original_segment, splitter
            )
        new_segments.extend(split_segment)
        new_delimiters.extend(split_delimiters)
        if i < len(segmented_text.delimiters):
            new_delimiters.append(segmented_text.delimiters[i])
    return SegmentedText(new_segments, new_delimiters)


def join_segments_to_next_by_condition(
    segmented_text: SegmentedText, checker: SegmentChecker
) -> SegmentedText:
    if len(segmented_text.segments) <= 1:
        return SegmentedText(segmented_text.segments, segmented_text.delimiters)
    reversed_segments = [segmented_text.segments[-1]]
    reversed_delimiters = []

    # Iterate through self.segments in reverse.
    # If a segment meets a condition (e.g., too small),
    # join it with the previous segment in reversed_segments(next segment in self.segments).
    for i in reversed(range(len(segmented_text.segments) - 1)):
        if checker(segmented_text.segments[i]):
            reversed_segments[-1] = join_with_delimiters(
                [segmented_text.segments[i], reversed_segments[-1]],
                [segmented_text.delimiters[i]],
            )
        else:
            reversed_segments.append(segmented_text.segments[i])
            reversed_delimiters.append(segmented_text.delimiters[i])

    # Handle the last segment (the first one in the reversed_segments) separately.
    if len(reversed_segments) > 1 and checker(reversed_segments[0]):
        reversed_segments[0:2] = [
            join_with_delimiters(
                [reversed_segments[1], reversed_segments[0]],
                [reversed_delimiters[0]],
            )
        ]
        reversed_delimiters[0:1] = []
    return SegmentedText(reversed_segments[::-1], reversed_delimiters[::-1])
