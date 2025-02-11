from aidial_rag_eval.metrics import calculate_inference, calculate_refusal


def test_inference(llm):
    inference_return = calculate_inference(
        "I am smart.", "I am not smart", llm, show_progress_bar=False
    )
    assert inference_return.inference == 0.0
    inference_return = calculate_inference(
        "I am smart.", "I am clever", llm, show_progress_bar=False
    )
    assert inference_return.inference == 1.0


def test_refusal(llm):
    refusal_return = calculate_refusal("I am smart.", llm, show_progress_bar=False)
    assert refusal_return.refusal == 0.0
    refusal_return = calculate_refusal(
        "there is no answer to this question", llm, show_progress_bar=False
    )
    assert refusal_return.refusal == 1.0
