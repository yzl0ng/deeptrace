from __future__ import annotations

import math

import pytest

from app.core.hybrid import HybridRetriever
from app.models import (
    DenseSearchHit,
    DenseSearchResponse,
    Document,
    SearchHit,
    SearchResponse,
)


DOCS = {
    name: Document(id=name, title=name, content=f"content for {name}")
    for name in ("doc-001", "doc-002", "doc-003", "doc-004")
}


class FixedLexicalRetriever:
    def __init__(self, rows: list[tuple[str, float]]) -> None:
        self.rows = rows

    def search(self, query: str, top_k: int) -> SearchResponse:
        hits = [
            SearchHit(
                rank=rank,
                score=score,
                document=DOCS[document_id],
                matched_terms=[],
                term_contributions=[],
            )
            for rank, (document_id, score) in enumerate(
                self.rows[:top_k], start=1
            )
        ]
        return SearchResponse(
            query=query,
            query_tokens=[],
            total_hits=len(self.rows),
            elapsed_ms=0.3,
            index_version=1,
            hits=hits,
        )


class FixedDenseRetriever:
    def __init__(self, rows: list[tuple[str, float]]) -> None:
        self.rows = rows

    def search(self, query: str, top_k: int) -> DenseSearchResponse:
        hits = [
            DenseSearchHit(
                rank=rank,
                score=score,
                document=DOCS[document_id],
            )
            for rank, (document_id, score) in enumerate(
                self.rows[:top_k], start=1
            )
        ]
        return DenseSearchResponse(
            query=query,
            model_name="fixed-vectors",
            device="cpu",
            vector_dimension=3,
            total_hits=len(self.rows),
            elapsed_ms=0.8,
            index_version=1,
            hits=hits,
        )


def make_retriever(
    bm25: list[tuple[str, float]],
    dense: list[tuple[str, float]],
) -> HybridRetriever:
    return HybridRetriever(
        FixedLexicalRetriever(bm25),
        FixedDenseRetriever(dense),
    )


def test_rrf_fuses_shared_and_single_path_documents_without_duplicates() -> None:
    retriever = make_retriever(
        [("doc-001", 100.0), ("doc-002", 2.0)],
        [("doc-002", 0.8), ("doc-003", 0.7)],
    )

    response = retriever.search("query", top_k=3)

    assert [hit.document.id for hit in response.hits] == [
        "doc-002",
        "doc-001",
        "doc-003",
    ]
    assert len({hit.document.id for hit in response.hits}) == 3
    shared = response.hits[0]
    assert shared.sources == ["bm25", "dense"]
    assert shared.source_ranks == {"bm25": 2, "dense": 1}
    assert math.isclose(
        shared.rrf_score,
        1 / 62 + 1 / 61,
        rel_tol=1e-12,
    )
    assert response.hits[1].sources == ["bm25"]
    assert response.hits[2].sources == ["dense"]


def test_rank_constant_changes_contributions() -> None:
    retriever = make_retriever([("doc-001", 1.0)], [("doc-001", 0.5)])

    response = retriever.search("query", top_k=1, rank_constant=10)

    assert response.rank_constant == 10
    assert response.hits[0].rrf_contributions == {
        "bm25": pytest.approx(1 / 11),
        "dense": pytest.approx(1 / 11),
    }


def test_candidate_k_must_not_be_smaller_than_top_k() -> None:
    retriever = make_retriever([], [])

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than or equal to top_k",
    ):
        retriever.search("query", top_k=3, candidate_k=2)


def test_ties_are_deterministic_and_use_document_id_last() -> None:
    retriever = make_retriever(
        [("doc-002", 9999.0)],
        [("doc-001", -9999.0)],
    )

    first = retriever.search("query", top_k=2)
    second = retriever.search("query", top_k=2)

    assert [hit.document.id for hit in first.hits] == ["doc-001", "doc-002"]
    assert [hit.document.id for hit in second.hits] == ["doc-001", "doc-002"]


def test_rrf_ignores_raw_score_scale() -> None:
    high_scale = make_retriever(
        [("doc-001", 1_000_000.0), ("doc-002", 1.0)],
        [("doc-002", 0.1), ("doc-001", 0.0)],
    )
    reversed_scale = make_retriever(
        [("doc-001", -1_000_000.0), ("doc-002", 999_999.0)],
        [("doc-002", -10.0), ("doc-001", 99.0)],
    )

    first = high_scale.search("query", top_k=2)
    second = reversed_scale.search("query", top_k=2)

    assert [hit.rrf_score for hit in first.hits] == [
        hit.rrf_score for hit in second.hits
    ]
    assert [hit.document.id for hit in first.hits] == [
        hit.document.id for hit in second.hits
    ]
