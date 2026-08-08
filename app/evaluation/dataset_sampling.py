from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


Qrel = tuple[str, str, int]


def select_query_ids(
    qrels: Iterable[Qrel],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Select judged queries deterministically from positive relevance labels."""
    if count < 1:
        raise ValueError("query count must be at least 1")
    candidates = sorted(
        {
            query_id
            for query_id, _, relevance in qrels
            if relevance > 0
        }
    )
    if count > len(candidates):
        raise ValueError(
            f"requested {count} queries, but only {len(candidates)} are judged"
        )
    selected = random.Random(seed).sample(candidates, count)
    return sorted(selected)


def select_document_ids(
    all_document_ids: Sequence[str],
    *,
    required_ids: Iterable[str],
    count: int,
    seed: int,
) -> list[str]:
    """Create a deterministic corpus containing every judged document."""
    if count < 1:
        raise ValueError("document count must be at least 1")
    available = set(all_document_ids)
    required = set(required_ids)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"{len(missing)} required documents are missing from the corpus"
        )
    if count < len(required):
        raise ValueError(
            f"document count {count} is smaller than "
            f"{len(required)} required documents"
        )
    if count > len(available):
        raise ValueError(
            f"requested {count} documents, but corpus has {len(available)}"
        )

    remaining = sorted(available - required)
    sampled = random.Random(seed).sample(
        remaining,
        count - len(required),
    )
    return sorted(required | set(sampled))


def filter_qrels(
    qrels: Iterable[Qrel],
    *,
    query_ids: Iterable[str],
    document_ids: Iterable[str],
) -> list[Qrel]:
    selected_queries = set(query_ids)
    selected_documents = set(document_ids)
    return sorted(
        (
            query_id,
            document_id,
            relevance,
        )
        for query_id, document_id, relevance in qrels
        if query_id in selected_queries
        and document_id in selected_documents
        and relevance > 0
    )
