from dataclasses import dataclass

from app.core.context_builder import ContextBuilder
from app.core.llm import LLMResult, LLMUsage
from app.core.rag import RAGService
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
        rrf_score=0.03,
        document=Document(
            id=document_id,
            title=f"Document {document_id}",
            content=f"Evidence from {document_id}",
            source="test",
        ),
        sources=["bm25", "dense"],
        source_ranks={"bm25": rank, "dense": rank},
        source_scores={"bm25": 1.0, "dense": 0.9},
        rrf_contributions={"bm25": 0.015, "dense": 0.015},
    )


class FakeHybridRetriever:
    def __init__(self, hits: list[HybridSearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int | None = None,
        rank_constant: int | None = None,
    ) -> HybridSearchResponse:
        self.calls += 1
        return HybridSearchResponse(
            query=query,
            rank_constant=rank_constant or 60,
            candidate_k=candidate_k or 20,
            elapsed_ms=1.0,
            retrieval_latency=RetrievalLatency(
                bm25_ms=0.2, dense_ms=0.7, fusion_ms=0.1
            ),
            dense_model=DenseModelInfo(
                name="fake-embedder", device="cpu", dimension=2
            ),
            hits=self.hits[:top_k],
        )


@dataclass
class FakeLLMClient:
    text: str
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    calls: int = 0
    last_context: str = ""

    def generate(self, *, query: str, context: str) -> LLMResult:
        self.calls += 1
        self.last_context = context
        return LLMResult(
            text=self.text,
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=40,
                completion_tokens=10,
                total_tokens=50,
            ),
        )


def service(answer: str, hits=None) -> tuple[RAGService, FakeLLMClient]:
    llm = FakeLLMClient(answer)
    return (
        RAGService(
            FakeHybridRetriever(
                hits
                if hits is not None
                else [make_hit(1, "doc-024"), make_hit(2, "doc-008")]
            ),
            llm,
            ContextBuilder(),
        ),
        llm,
    )


def test_model_called_once_and_usage_is_not_estimated() -> None:
    rag, llm = service("回答 [doc-024]")
    response = rag.answer("问题")

    assert llm.calls == 1
    assert response.usage.model_dump() == {
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
        "estimated_cost": None,
        "currency": None,
    }


def test_no_context_does_not_call_model() -> None:
    rag, llm = service("must not be used", hits=[])
    response = rag.answer("问题")

    assert llm.calls == 0
    assert response.abstained is True
    assert response.abstention_reason == "no_retrieval_context"
    assert response.latency.generation_ms == 0
    assert response.citations == []


def test_parses_multiple_citations_deduplicates_and_preserves_first_order() -> None:
    rag, _ = service(
        "事实 A [doc-008]，事实 B [doc-024]，再次引用 [doc-008]。"
    )
    response = rag.answer("问题")

    assert [item.citation_id for item in response.citations] == [
        "doc-008",
        "doc-024",
    ]


def test_fabricated_citation_is_explicitly_invalid() -> None:
    rag, _ = service("事实 [doc-024]，伪造 [doc-999]。")
    response = rag.answer("问题")

    assert [item.citation_id for item in response.citations] == ["doc-024"]
    assert response.invalid_citation_ids == ["doc-999"]


def test_citation_cannot_map_outside_actual_context_after_truncation() -> None:
    hits = [make_hit(index, f"doc-{index:03d}") for index in range(1, 7)]
    rag, _ = service("越界引用 [doc-006]", hits=hits)
    response = rag.answer("问题", retrieval_top_k=5)

    assert response.citations == []
    assert response.invalid_citation_ids == ["doc-006"]


def test_model_abstention_marker_is_removed_from_user_answer() -> None:
    rag, _ = service("INSUFFICIENT_EVIDENCE: 缺少直接证据。")
    response = rag.answer("问题")

    assert response.answer == "缺少直接证据。"
    assert response.abstained is True
    assert response.abstention_reason == "insufficient_evidence"
