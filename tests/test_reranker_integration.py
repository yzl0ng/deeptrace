import os

import pytest

from app.core.reranker import RerankerRuntime, RerankerSettings
from app.models import Document, HybridSearchHit


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_RERANKER_INTEGRATION") != "1",
        reason=(
            "Set RUN_RERANKER_INTEGRATION=1 to allow loading or downloading "
            "the real BGE reranker."
        ),
    ),
]


def make_hit(rank: int, document: Document) -> HybridSearchHit:
    return HybridSearchHit(
        rank=rank,
        rrf_score=1.0 / (60 + rank),
        document=document,
        sources=["bm25", "dense"],
        source_ranks={"bm25": rank, "dense": rank},
        source_scores={"bm25": 1.0, "dense": 0.9},
        rrf_contributions={"bm25": 0.01, "dense": 0.01},
    )


def test_real_bge_reranker_promotes_direct_evidence() -> None:
    runtime = RerankerRuntime(
        RerankerSettings.from_environment()
    )
    hits = [
        make_hit(
            1,
            Document(
                id="doc-bm25",
                title="BM25",
                content="BM25 uses term frequency and document frequency.",
            ),
        ),
        make_hit(
            2,
            Document(
                id="doc-grounding",
                title="可信回答与拒答",
                content="知识库证据不足时系统应该拒绝回答，避免模型编造事实。",
            ),
        ),
        make_hit(
            3,
            Document(
                id="doc-hnsw",
                title="HNSW",
                content="HNSW is an approximate nearest-neighbor index.",
            ),
        ),
    ]

    result = runtime.rerank(
        "知识库没有证据时，怎样避免大模型编造答案？",
        hits,
        top_k=3,
    )

    assert result.hits[0].document.id == "doc-grounding"
    assert result.model.name == RerankerSettings.from_environment().model_name
    assert result.elapsed_ms >= 0
