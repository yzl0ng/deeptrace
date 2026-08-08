from __future__ import annotations

import pytest

from app.main import hybrid_retriever


@pytest.mark.integration
def test_real_bge_dense_and_hybrid() -> None:
    response = hybrid_retriever.search(
        "如何避免模型产生幻觉",
        top_k=5,
        candidate_k=8,
    )

    assert response.dense_model.name == "BAAI/bge-m3"
    assert response.dense_model.device in {"cpu", "cuda"}
    assert response.dense_model.dimension == 1024
    assert len({hit.document.id for hit in response.hits}) == len(response.hits)
    assert "doc-008" in {hit.document.id for hit in response.hits}
    assert all(hit.rrf_score > 0 for hit in response.hits)
