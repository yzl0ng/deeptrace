from dataclasses import dataclass

import numpy as np

from app.core.context_builder import ContextBuilder
from app.core.llm import LLMResult, LLMUsage
from app.core.rag import RAGService
from app.core.reranker import CrossEncoderReranker
from app.models import (
    DenseModelInfo,
    Document,
    HybridSearchHit,
    HybridSearchResponse,
    RetrievalLatency,
)


def make_hit(rank: int, document_id: str) -> HybridSearchHit:
    return HybridSearchHit(
        rank=rank,
        rrf_score=1.0 / (60 + rank),
        document=Document(
            id=document_id,
            title=f"Document {document_id}",
            content=f"Evidence from {document_id}",
            source="test",
        ),
        sources=["bm25", "dense"],
        source_ranks={"bm25": rank, "dense": rank},
        source_scores={"bm25": 1.0, "dense": 0.9},
        rrf_contributions={"bm25": 0.01, "dense": 0.01},
    )


class CandidateRetriever:
    def __init__(self, hits: list[HybridSearchHit]) -> None:
        self.hits = hits
        self.requested_top_k: int | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int | None = None,
        rank_constant: int | None = None,
    ) -> HybridSearchResponse:
        self.requested_top_k = top_k
        return HybridSearchResponse(
            query=query,
            rank_constant=rank_constant or 60,
            candidate_k=candidate_k or 20,
            elapsed_ms=1.0,
            retrieval_latency=RetrievalLatency(
                bm25_ms=0.2,
                dense_ms=0.7,
                fusion_ms=0.1,
            ),
            dense_model=DenseModelInfo(
                name="fake-embedder",
                device="cpu",
                dimension=2,
            ),
            hits=self.hits[:top_k],
        )


class FixedPairScorer:
    model_name = "fake-reranker"
    device = "cpu"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        return np.asarray([0.1, 0.2, 4.0], dtype=np.float32)


@dataclass
class RecordingLLM:
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    calls: int = 0
    context: str = ""

    def generate(self, *, query: str, context: str) -> LLMResult:
        self.calls += 1
        self.context = context
        return LLMResult(
            text="第三篇证据最相关 [doc-003]。",
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
            system_prompt="grounded system",
            user_prompt="grounded user",
        )


def test_reranked_rag_preserves_baseline_rrf_rank_and_new_rank() -> None:
    retriever = CandidateRetriever(
        [make_hit(1, "doc-001"), make_hit(2, "doc-002"), make_hit(3, "doc-003")]
    )
    llm = RecordingLLM()
    service = RAGService(
        retriever,
        llm,
        ContextBuilder(),
        reranker=CrossEncoderReranker(FixedPairScorer()),
    )

    response = service.answer(
        "question",
        retrieval_top_k=2,
        candidate_k=3,
        rank_constant=60,
    )

    assert retriever.requested_top_k == 3
    assert response.retrieval.method == "rrf_reranker"
    assert [hit.document.id for hit in response.retrieval.hits] == [
        "doc-003",
        "doc-002",
    ]
    assert response.retrieval.reranking is not None
    assert response.retrieval.reranking.candidate_count == 3
    assert response.retrieval.reranking.output_count == 2
    assert response.latency.reranker_ms is not None
    assert response.citations[0].rrf_rank == 3
    assert response.citations[0].rerank_rank == 1
    assert llm.context.index("[DOC doc-003]") < llm.context.index("[DOC doc-002]")
    assert "Rerank Rank: 1\nRRF Rank: 3" in llm.context


def test_baseline_rag_still_requests_only_final_top_k() -> None:
    retriever = CandidateRetriever(
        [make_hit(1, "doc-001"), make_hit(2, "doc-002"), make_hit(3, "doc-003")]
    )
    llm = RecordingLLM()
    service = RAGService(retriever, llm, ContextBuilder())

    response = service.answer(
        "question",
        retrieval_top_k=2,
        candidate_k=3,
    )

    assert retriever.requested_top_k == 2
    assert response.retrieval.method == "rrf"
    assert response.retrieval.reranking is None
    assert response.latency.reranker_ms is None
