"""2026-10-10: explicit denominators and reproducible descriptive metrics."""
import math


def percentile(values, percent):
    """Nearest-rank percentile; empty observations are unknown, never zero."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def rate(successes, total):
    if not 0 <= successes <= total:
        raise ValueError("invalid numerator/denominator")
    if not total:
        return {"numerator": successes, "denominator": total, "value": None, "wilson95": None}
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"numerator": successes, "denominator": total, "value": p,
            "wilson95": [max(0, center - half), min(1, center + half)]}


def retrieval_scores(relevant_ids, retrieved_ids, k):
    """Binary relevance; duplicate hits cannot inflate recall or nDCG."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None}
    seen, hits = set(), []
    for index, item in enumerate(retrieved_ids[:k]):
        if item in relevant and item not in seen:
            hits.append(index + 1)
        seen.add(item)
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {"recall_at_k": len(hits) / len(relevant),
            "mrr_at_k": 1 / hits[0] if hits else 0.0, "ndcg_at_k": dcg / ideal}
