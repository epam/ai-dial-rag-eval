import numpy as np

from rag_eval_metrics.types import ContextRelevance, FactsRanks


def calculate_recall(facts_ranks: FactsRanks) -> np.float64:
    total_facts = len(facts_ranks)
    relevant_facts = np.count_nonzero(facts_ranks >= 0)
    recall = np.float64(relevant_facts) / np.float64(total_facts)
    return recall


def calculate_precision(context_relevance: ContextRelevance) -> np.float64:
    total_context = len(context_relevance)
    if total_context == 0:
        return np.float64(0.0)
    relevant_context = np.count_nonzero(context_relevance > 0)
    precision = np.float64(relevant_context) / np.float64(total_context)
    return precision


def calculate_f1(precision: np.float64, recall: np.float64) -> np.float64:
    if precision == 0 and recall == 0:
        return np.float64(0.0)
    else:
        return 2.0 * np.float64(precision * recall) / np.float64(precision + recall)


def calculate_mrr(facts_ranks: FactsRanks) -> np.floating:
    ranks = facts_ranks.astype(float) + 1
    ranks[ranks == 0] = np.inf
    mrr = np.mean(1 / ranks)
    return mrr
