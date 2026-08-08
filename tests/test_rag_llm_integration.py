import os

import pytest

from app.core.context_builder import ContextBuilder
from app.core.llm import DeepSeekClient, DeepSeekSettings
from app.core.rag import RAGService
from app.main import hybrid_retriever


@pytest.mark.integration
@pytest.mark.llm_integration
def test_real_deepseek_grounded_answer() -> None:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        pytest.skip("DEEPSEEK_API_KEY is not configured")

    settings = DeepSeekSettings.from_environment()
    service = RAGService(
        hybrid_retriever,
        DeepSeekClient(settings),
        ContextBuilder(),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000")),
    )
    response = service.answer(
        "如何避免模型产生幻觉",
        retrieval_top_k=5,
        candidate_k=20,
        rank_constant=60,
    )

    retrieved_ids = {
        hit.document.id for hit in response.retrieval.hits
    }
    assert response.model.provider == "deepseek"
    assert response.model.name
    assert response.usage.total_tokens > 0
    assert response.latency.generation_ms > 0
    assert response.citations
    assert all(
        citation.citation_id in retrieved_ids
        for citation in response.citations
    )
    assert not response.invalid_citation_ids
