import numpy as np
import pytest

from app.core.reranker import (
    CrossEncoderReranker,
    RerankerRuntime,
    RerankerSettings,
    RerankerUnavailableError,
    select_reranker_device,
)
from app.models import Document, HybridSearchHit


def make_hit(rank: int, document_id: str) -> HybridSearchHit:
    return HybridSearchHit(
        rank=rank,
        rrf_score=1.0 / (60 + rank),
        document=Document(
            id=document_id,
            title=f"Title {document_id}",
            content=f"Content {document_id}",
        ),
        sources=["bm25", "dense"],
        source_ranks={"bm25": rank, "dense": rank},
        source_scores={"bm25": 1.0, "dense": 0.8},
        rrf_contributions={"bm25": 0.01, "dense": 0.01},
    )


class FakePairScorer:
    model_name = "fake-cross-encoder"
    device = "cpu"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0
        self.last_query = ""
        self.last_documents: list[str] = []

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        self.calls += 1
        self.last_query = query
        self.last_documents = list(documents)
        return np.asarray(self.scores, dtype=np.float32)


def test_reranker_reorders_candidates_and_preserves_rrf_trace() -> None:
    scorer = FakePairScorer([0.1, -0.2, 3.5])
    reranker = CrossEncoderReranker(scorer)

    result = reranker.rerank(
        "query",
        [make_hit(1, "doc-001"), make_hit(2, "doc-002"), make_hit(3, "doc-003")],
        top_k=2,
    )

    assert [hit.document.id for hit in result.hits] == ["doc-003", "doc-001"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.hits[0].rrf_rank == 3
    assert result.hits[0].rank_delta == 2
    assert result.hits[0].reranker_score == pytest.approx(3.5)
    assert result.candidate_count == 3
    assert len(result.traces) == 3
    moved = next(
        trace for trace in result.traces if trace.document_id == "doc-003"
    )
    assert moved.document_title == "Title doc-003"
    assert moved.rrf_rank == 3
    assert moved.rerank_rank == 1
    assert moved.rank_delta == 2
    assert moved.reranker_score == pytest.approx(3.5)
    assert scorer.last_query == "query"
    assert scorer.last_documents[2] == "Title doc-003\nContent doc-003"


def test_equal_scores_keep_original_rrf_order() -> None:
    reranker = CrossEncoderReranker(FakePairScorer([1.0, 1.0, 1.0]))

    result = reranker.rerank(
        "query",
        [make_hit(1, "doc-b"), make_hit(2, "doc-a"), make_hit(3, "doc-c")],
        top_k=3,
    )

    assert [hit.document.id for hit in result.hits] == [
        "doc-b",
        "doc-a",
        "doc-c",
    ]


def test_all_candidates_are_traced_when_only_top_k_are_returned() -> None:
    result = CrossEncoderReranker(
        FakePairScorer([0.4, 0.3, 0.2])
    ).rerank(
        "query",
        [make_hit(1, "doc-1"), make_hit(2, "doc-2"), make_hit(3, "doc-3")],
        top_k=1,
    )

    assert len(result.hits) == 1
    assert len(result.traces) == 3


def test_empty_candidates_do_not_call_pair_scorer() -> None:
    scorer = FakePairScorer([])
    result = CrossEncoderReranker(scorer).rerank("query", [], top_k=5)

    assert result.hits == []
    assert result.traces == []
    assert scorer.calls == 0


def test_invalid_scorer_output_is_rejected() -> None:
    reranker = CrossEncoderReranker(FakePairScorer([1.0]))

    with pytest.raises(ValueError, match="one score per candidate"):
        reranker.rerank(
            "query",
            [make_hit(1, "doc-1"), make_hit(2, "doc-2")],
            top_k=1,
        )


def test_runtime_loads_pair_scorer_only_once() -> None:
    created: list[FakePairScorer] = []

    def factory(_: RerankerSettings) -> FakePairScorer:
        scorer = FakePairScorer([1.0])
        created.append(scorer)
        return scorer

    runtime = RerankerRuntime(
        RerankerSettings(model_name="fake"),
        scorer_factory=factory,
    )
    hits = [make_hit(1, "doc-1")]

    runtime.rerank("first", hits, 1)
    runtime.rerank("second", hits, 1)

    assert len(created) == 1
    assert created[0].calls == 2


def test_reranker_device_selection_is_explicit() -> None:
    assert select_reranker_device("auto", cuda_available=True) == "cuda"
    assert select_reranker_device("auto", cuda_available=False) == "cpu"
    assert select_reranker_device("cpu", cuda_available=True) == "cpu"
    with pytest.raises(RerankerUnavailableError, match="CUDA is not available"):
        select_reranker_device("cuda", cuda_available=False)
