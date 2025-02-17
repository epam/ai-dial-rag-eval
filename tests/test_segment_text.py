from aidial_rag_eval.generation.utils.segmented_text import SegmentedText


def test_segment_text():
    segmented_text = SegmentedText.from_text(
        "Very important hypotheses to list."
        "\n\t1. Hypothesis number one. 2. Hypothesis number two."
    )
    assert segmented_text.segments == [
        "Very important hypotheses to list.",
        "1. Hypothesis number one.",
        "2. Hypothesis number two.",
    ]
    assert segmented_text.delimiters == ["\n\t", " "]


def test_segment_text_short():
    segmented_text = SegmentedText.from_text("A short sentence at the end. Short.")
    assert segmented_text.segments == ["A short sentence at the end. Short."]
    assert segmented_text.delimiters == []
