from aidial_rag_eval.generation.metric_binds import metric_bind_keys

CONTEXT_TO_ANSWER_INFERENCE = metric_bind_keys[0]
ANSWER_TO_GROUND_TRUTH_INFERENCE = metric_bind_keys[1]
GROUND_TRUTH_TO_ANSWER_INFERENCE = metric_bind_keys[2]
ANSWER_REFUSAL = metric_bind_keys[3]
GROUND_TRUTH_REFUSAL = metric_bind_keys[4]

__all__ = [
    "CONTEXT_TO_ANSWER_INFERENCE",
    "ANSWER_TO_GROUND_TRUTH_INFERENCE",
    "GROUND_TRUTH_TO_ANSWER_INFERENCE",
    "ANSWER_REFUSAL",
    "GROUND_TRUTH_REFUSAL",
]
