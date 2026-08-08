from fastapi.testclient import TestClient
import numpy as np

from app.core.context_builder import ContextBuilder
from app.core.llm import LLMResult, LLMUsage
from app.core.rag import RAGService
from app.core.reranker import CrossEncoderReranker
from app.main import app, get_reranked_rag_service
from app.models import (
    DenseModelInfo,
    Document,
    HybridSearchHit,
    HybridSearchResponse,
    RetrievalLatency,
)


client = TestClient(app)


class ApiCandidateRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int | None = None,
        rank_constant: int | None = None,
    ) -> HybridSearchResponse:
        hits = [
            HybridSearchHit(
                rank=rank,
                rrf_score=1.0 / (60 + rank),
                document=Document(
                    id=f"doc-{rank:03d}",
                    title=f"Document {rank}",
                    content=f"Evidence {rank}",
                ),
                sources=["bm25", "dense"],
                source_ranks={"bm25": rank, "dense": rank},
                source_scores={"bm25": 1.0, "dense": 0.9},
                rrf_contributions={"bm25": 0.01, "dense": 0.01},
            )
            for rank in range(1, 4)
        ]
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
                name="fake",
                device="cpu",
                dimension=2,
            ),
            hits=hits[:top_k],
        )


class ApiPairScorer:
    model_name = "fake-reranker"
    device = "cpu"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        return np.asarray([0.1, 0.2, 5.0], dtype=np.float32)


class ApiLLM:
    provider = "deepseek"
    model_name = "deepseek-chat"

    def generate(self, *, query: str, context: str) -> LLMResult:
        return LLMResult(
            text="重排后的回答 [doc-003]。",
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
            system_prompt="system",
            user_prompt="user",
        )


def test_reranked_rag_api_contract() -> None:
    service = RAGService(
        ApiCandidateRetriever(),
        ApiLLM(),
        ContextBuilder(),
        reranker=CrossEncoderReranker(ApiPairScorer()),
    )
    app.dependency_overrides[get_reranked_rag_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/rag/reranked-answer",
            json={
                "query": "question",
                "retrieval_top_k": 2,
                "candidate_k": 3,
                "rank_constant": 60,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["method"] == "rrf_reranker"
    assert payload["retrieval"]["reranking"]["model"] == {
        "name": "fake-reranker",
        "device": "cpu",
    }
    assert payload["retrieval"]["hits"][0]["document"]["id"] == "doc-003"
    assert payload["retrieval"]["reranking"]["traces"][0]["document_title"]
    assert payload["citations"][0]["rrf_rank"] == 3
    assert payload["citations"][0]["rerank_rank"] == 1
    assert payload["latency"]["reranker_ms"] >= 0
