from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.evaluation.corpus_audit import percentile
from app.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> Any: ...


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    category: str
    relevance: Mapping[str, int]
    reviewed: bool


def evaluate_retriever(
    name: str,
    retriever: Retriever,
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not queries:
        raise ValueError("evaluation requires at least one query")
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in queries:
        started_at = time.perf_counter()
        response = retriever.search(item.query, top_k)
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies.append(latency_ms)
        ranked_ids = [hit.document.id for hit in response.hits]
        scores = [
            float(
                hit.score
                if hasattr(hit, "score")
                else hit.rrf_score
            )
            for hit in response.hits
        ]
        metrics = compute_query_metrics(
            ranked_ids,
            item.relevance,
        )
        per_query.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "category": item.category,
                "retriever": name,
                "reviewed": item.reviewed,
                "latency_ms": latency_ms,
                "ranked_chunk_ids": ranked_ids,
                "scores": scores,
                "relevance": dict(item.relevance),
                "metrics": metrics,
            }
        )

    metric_names = list(per_query[0]["metrics"])
    summary = {
        "queries": len(queries),
        **{
            metric: statistics.fmean(
                row["metrics"][metric] for row in per_query
            )
            for metric in metric_names
        },
        "latency": latency_summary(latencies),
    }
    return summary, per_query


def compute_query_metrics(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_ids, relevance, 1),
        "recall@3": recall_at_k(ranked_ids, relevance, 3),
        "recall@5": recall_at_k(ranked_ids, relevance, 5),
        "recall@10": recall_at_k(ranked_ids, relevance, 10),
        "precision@5": precision_at_k(ranked_ids, relevance, 5),
        "precision@10": precision_at_k(ranked_ids, relevance, 10),
        "mrr": reciprocal_rank(ranked_ids, relevance),
        "ndcg@5": ndcg_at_k(ranked_ids, relevance, 5),
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
    }


def latency_summary(latencies: Sequence[float]) -> dict[str, float | int]:
    if not latencies:
        return {
            "queries": 0,
            "average_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "queries": len(latencies),
        "average_ms": statistics.fmean(latencies),
        "p50_ms": percentile(list(latencies), 0.50),
        "p95_ms": percentile(list(latencies), 0.95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
