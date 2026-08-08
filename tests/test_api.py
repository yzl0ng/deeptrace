from fastapi.testclient import TestClient
import numpy as np

from app.core.dense import DenseRuntime, DenseSettings, DenseUnavailableError
from app.core.hybrid import HybridRetriever
from app.core.context_builder import ContextBuilder
from app.core.llm import LLMResult, LLMUsage
from app.core.rag import RAGService
from app.main import (
    app,
    get_dense_runtime,
    get_hybrid_retriever,
    get_llm_client,
    get_rag_service,
)
from app.models import Document


client = TestClient(app)


def test_health_exposes_engine_and_index_size() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "stage": "bm25-baseline",
        "engine": "python-bm25",
        "documents": 30,
    }


def test_search_contract_contains_explanations() -> None:
    response = client.get(
        "/api/v1/search",
        params={"q": "RRF 如何融合结果", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"][0]["document"]["id"] == "doc-003"
    assert payload["hits"][0]["term_contributions"]
    assert payload["elapsed_ms"] >= 0
    assert payload["index_version"] == 1


def test_index_stats_contract() -> None:
    response = client.get("/api/v1/index/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"] == 30
    assert payload["unique_terms"] > 0
    assert payload["k1"] == 1.5
    assert payload["b"] == 0.75


def test_latest_evaluation_returns_saved_real_report() -> None:
    response = client.get("/api/v1/evaluation/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["corpus_version"] == "quality-v1"
    assert payload["config"]["embedding_model"] == "BAAI/bge-m3"
    assert payload["config"]["embedding_cache"]["cache_hit"] is True
    assert payload["summary"]["dense_exact"]["recall@10"] >= 0


def test_no_match_is_a_successful_empty_result() -> None:
    response = client.get(
        "/api/v1/search",
        params={"q": "quantum-teleportation", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["hits"] == []
    assert response.json()["total_hits"] == 0


def test_cors_allows_local_web_app() -> None:
    response = client.options(
        "/api/v1/search?q=BM25",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_rag_api_reports_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    response = client.post(
        "/api/v1/rag/answer",
        json={"query": "如何避免模型产生幻觉"},
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "rag_not_configured",
            "message": "DEEPSEEK_API_KEY is not configured.",
        }
    }


def test_plain_llm_api_reports_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    response = client.post(
        "/api/v1/llm/answer",
        json={"query": "RRF 是什么"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_not_configured"


class ApiFakeEmbedder:
    model_name = "fake-api-embedder"
    device = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "BM25 关键词检索\n关键词精确匹配": [1.0, 0.0],
            "可信回答与引用\n证据不足时拒绝回答": [0.0, 1.0],
            "如何避免模型产生幻觉": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_dense_search_api_contract() -> None:
    runtime = DenseRuntime(
        documents=[
            Document(
                id="doc-001",
                title="BM25 关键词检索",
                content="关键词精确匹配",
            ),
            Document(
                id="doc-008",
                title="可信回答与引用",
                content="证据不足时拒绝回答",
            ),
        ],
        settings=DenseSettings(model_name="fake-api-embedder"),
        embedder_factory=lambda _: ApiFakeEmbedder(),
    )
    app.dependency_overrides[get_dense_runtime] = lambda: runtime
    try:
        response = client.get(
            "/api/v1/search/dense",
            params={"q": "如何避免模型产生幻觉", "top_k": 1},
        )
        stats_response = client.get("/api/v1/index/dense/stats")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "fake-api-embedder"
    assert payload["device"] == "cpu"
    assert payload["vector_dimension"] == 2
    assert payload["hits"][0]["document"]["id"] == "doc-008"
    assert stats_response.json()["status"] == "ready"


def test_dense_api_reports_model_unavailability() -> None:
    def unavailable_factory(_: DenseSettings) -> ApiFakeEmbedder:
        raise DenseUnavailableError("model files are unavailable")

    runtime = DenseRuntime(
        documents=[],
        settings=DenseSettings(model_name="missing-model"),
        embedder_factory=unavailable_factory,
    )
    app.dependency_overrides[get_dense_runtime] = lambda: runtime
    try:
        response = client.get(
            "/api/v1/search/dense",
            params={"q": "test", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "dense_unavailable",
        "message": "model files are unavailable",
        "status": "error",
    }


def test_hybrid_search_api_contract() -> None:
    runtime = DenseRuntime(
        documents=[
            Document(id="doc-001", title="BM25", content="keyword retrieval"),
            Document(id="doc-008", title="可信回答", content="证据不足时拒答"),
        ],
        settings=DenseSettings(model_name="fake-api-embedder"),
        embedder_factory=lambda _: ApiHybridEmbedder(),
    )
    lexical = ApiLexicalRetriever()
    retriever = HybridRetriever(lexical, runtime)
    app.dependency_overrides[get_hybrid_retriever] = lambda: retriever
    try:
        response = client.get(
            "/api/v1/search/hybrid",
            params={
                "q": "如何避免模型产生幻觉",
                "top_k": 2,
                "candidate_k": 2,
                "rank_constant": 60,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["dense_model"] == {
        "name": "fake-api-embedder",
        "device": "cpu",
        "dimension": 2,
    }
    assert {hit["document"]["id"] for hit in payload["hits"]} == {
        "doc-001",
        "doc-008",
    }
    doc_008 = next(
        hit for hit in payload["hits"] if hit["document"]["id"] == "doc-008"
    )
    assert doc_008["sources"] == ["bm25", "dense"]
    assert doc_008["source_ranks"] == {"bm25": 2, "dense": 1}
    assert set(payload["retrieval_latency"]) == {
        "bm25_ms",
        "dense_ms",
        "fusion_ms",
    }


def test_hybrid_api_validates_candidate_k() -> None:
    response = client.get(
        "/api/v1/search/hybrid",
        params={"q": "test", "top_k": 3, "candidate_k": 2},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_candidate_k"


def test_hybrid_api_reports_dense_model_failure() -> None:
    def unavailable_factory(_: DenseSettings) -> ApiFakeEmbedder:
        raise DenseUnavailableError("Unable to load embedding model")

    runtime = DenseRuntime(
        documents=[],
        settings=DenseSettings(model_name="missing-model"),
        embedder_factory=unavailable_factory,
    )
    retriever = HybridRetriever(ApiLexicalRetriever(), runtime)
    app.dependency_overrides[get_hybrid_retriever] = lambda: retriever
    try:
        response = client.get(
            "/api/v1/search/hybrid",
            params={"q": "test", "top_k": 1, "candidate_k": 1},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_load_failed"


class ApiHybridEmbedder:
    model_name = "fake-api-embedder"
    device = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "BM25\nkeyword retrieval": [1.0, 0.0],
            "可信回答\n证据不足时拒答": [0.0, 1.0],
            "如何避免模型产生幻觉": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class ApiLexicalRetriever:
    def search(self, query: str, top_k: int):
        from app.models import SearchHit, SearchResponse

        docs = [
            Document(id="doc-001", title="BM25", content="keyword retrieval"),
            Document(id="doc-008", title="可信回答", content="证据不足时拒答"),
        ]
        return SearchResponse(
            query=query,
            query_tokens=[],
            total_hits=2,
            elapsed_ms=0.1,
            index_version=1,
            hits=[
                SearchHit(
                    rank=rank,
                    score=float(3 - rank),
                    document=document,
                    matched_terms=[],
                    term_contributions=[],
                )
                for rank, document in enumerate(docs[:top_k], start=1)
            ],
        )


class ApiFakeLLM:
    provider = "deepseek"
    model_name = "deepseek-chat"

    def generate(self, *, query: str, context: str) -> LLMResult:
        assert "[DOC doc-008]" in context
        return LLMResult(
            text="应在证据不足时拒答 [doc-008]。",
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
            system_prompt="grounded system prompt",
            user_prompt="question plus retrieved context",
        )

    def generate_plain(self, *, query: str) -> LLMResult:
        return LLMResult(
            text="这是没有检索上下文的普通回答。",
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=9,
                completion_tokens=7,
                total_tokens=16,
            ),
            system_prompt="plain system prompt",
            user_prompt=query,
        )


def test_rag_api_contract() -> None:
    runtime = DenseRuntime(
        documents=[
            Document(id="doc-001", title="BM25", content="keyword retrieval"),
            Document(id="doc-008", title="可信回答", content="证据不足时拒答"),
        ],
        settings=DenseSettings(model_name="fake-api-embedder"),
        embedder_factory=lambda _: ApiHybridEmbedder(),
    )
    service = RAGService(
        HybridRetriever(ApiLexicalRetriever(), runtime),
        ApiFakeLLM(),
        ContextBuilder(),
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/rag/answer",
            json={
                "query": "如何避免模型产生幻觉",
                "retrieval_top_k": 2,
                "candidate_k": 2,
                "rank_constant": 60,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == {
        "provider": "deepseek",
        "name": "deepseek-chat",
    }
    assert payload["retrieval"]["method"] == "rrf"
    assert payload["retrieval"]["top_k"] == 2
    assert payload["citations"][0]["citation_id"] == "doc-008"
    assert payload["usage"]["total_tokens"] == 28
    assert payload["usage"]["estimated_cost"] is None
    assert payload["latency"]["generation_ms"] >= 0
    assert payload["prompt"]["system"] == "grounded system prompt"
    assert payload["prompt"]["user"] == "question plus retrieved context"


def test_plain_llm_api_contract_has_no_retrieval_or_citations() -> None:
    app.dependency_overrides[get_llm_client] = lambda: ApiFakeLLM()
    try:
        response = client.post(
            "/api/v1/llm/answer",
            json={"query": "RRF 是什么"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "这是没有检索上下文的普通回答。"
    assert payload["model"] == {
        "provider": "deepseek",
        "name": "deepseek-chat",
    }
    assert payload["retrieval_used"] is False
    assert payload["citations"] == []
    assert payload["usage"]["total_tokens"] == 16
    assert payload["usage"]["estimated_cost"] is None
    assert payload["latency"]["generation_ms"] >= 0
    assert payload["prompt"]["system"] == "plain system prompt"
    assert payload["prompt"]["user"] == payload["query"]
