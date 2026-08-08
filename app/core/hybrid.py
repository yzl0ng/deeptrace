from __future__ import annotations

import time
from typing import Protocol

from app.models import (
    DenseModelInfo,
    DenseSearchResponse,
    Document,
    HybridSearchHit,
    HybridSearchResponse,
    RetrievalLatency,
    SearchResponse,
)


class LexicalRetriever(Protocol):
    def search(self, query: str, top_k: int) -> SearchResponse: ...


class SemanticRetriever(Protocol):
    def search(self, query: str, top_k: int) -> DenseSearchResponse: ...


class HybridRetriever:
    """Fuse existing lexical and dense ranks with Reciprocal Rank Fusion."""

    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        dense_retriever: SemanticRetriever,
        rank_constant: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rank_constant <= 0:
            raise ValueError("rank_constant must be greater than 0")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.rank_constant = rank_constant
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int | None = None,
        rank_constant: int | None = None,
    ) -> HybridSearchResponse:
        effective_candidate_k = (
            self.candidate_k if candidate_k is None else candidate_k
        )
        effective_rank_constant = (
            self.rank_constant if rank_constant is None else rank_constant
        )
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if effective_candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if effective_rank_constant <= 0:
            raise ValueError("rank_constant must be greater than 0")

        started_at = time.perf_counter()
        lexical_response = self.lexical_retriever.search(
            query, effective_candidate_k
        )
        dense_response = self.dense_retriever.search(
            query, effective_candidate_k
        )

        fusion_started_at = time.perf_counter()
        candidates: dict[str, dict[str, object]] = {}
        self._add_candidates(
            candidates,
            lexical_response.hits,
            source="bm25",
            rank_constant=effective_rank_constant,
        )
        self._add_candidates(
            candidates,
            dense_response.hits,
            source="dense",
            rank_constant=effective_rank_constant,
        )

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item["rrf_score"]),
                -len(item["sources"]),
                min(item["source_ranks"].values()),
                item["document"].id,
            ),
        )
        hits = [
            HybridSearchHit(
                rank=rank,
                rrf_score=float(item["rrf_score"]),
                document=item["document"],
                sources=item["sources"],
                source_ranks=item["source_ranks"],
                source_scores=item["source_scores"],
                rrf_contributions=item["rrf_contributions"],
            )
            for rank, item in enumerate(ordered[:top_k], start=1)
        ]
        fusion_ms = (time.perf_counter() - fusion_started_at) * 1000
        return HybridSearchResponse(
            query=query,
            rank_constant=effective_rank_constant,
            candidate_k=effective_candidate_k,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            retrieval_latency=RetrievalLatency(
                bm25_ms=lexical_response.elapsed_ms,
                dense_ms=dense_response.elapsed_ms,
                fusion_ms=fusion_ms,
            ),
            dense_model=DenseModelInfo(
                name=dense_response.model_name,
                device=dense_response.device,
                dimension=dense_response.vector_dimension,
            ),
            hits=hits,
        )

    @staticmethod
    def _add_candidates(
        candidates: dict[str, dict[str, object]],
        hits: list,
        *,
        source: str,
        rank_constant: int,
    ) -> None:
        for hit in hits:
            document: Document = hit.document
            contribution = 1.0 / (rank_constant + hit.rank)
            candidate = candidates.setdefault(
                document.id,
                {
                    "document": document,
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "rrf_contributions": {},
                    "rrf_score": 0.0,
                },
            )
            if source in candidate["source_ranks"]:
                continue
            candidate["sources"].append(source)
            candidate["source_ranks"][source] = hit.rank
            candidate["source_scores"][source] = hit.score
            candidate["rrf_contributions"][source] = contribution
            candidate["rrf_score"] += contribution
