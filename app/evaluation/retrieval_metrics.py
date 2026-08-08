from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    _validate_k(k)
    relevant = {item_id for item_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def precision_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    _validate_k(k)
    relevant = {item_id for item_id, grade in relevance.items() if grade > 0}
    hits = sum(item_id in relevant for item_id in ranked_ids[:k])
    return hits / k


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
) -> float:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if relevance.get(item_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_grades: Sequence[int], k: int) -> float:
    _validate_k(k)
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(relevance_grades[:k], start=1)
        if grade > 0
    )


def idcg_at_k(relevance: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    ideal = sorted(
        (grade for grade in relevance.values() if grade > 0),
        reverse=True,
    )
    return dcg_at_k(ideal, k)


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    ideal = idcg_at_k(relevance, k)
    if ideal == 0:
        return 0.0
    observed = [relevance.get(item_id, 0) for item_id in ranked_ids]
    return dcg_at_k(observed, k) / ideal


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
