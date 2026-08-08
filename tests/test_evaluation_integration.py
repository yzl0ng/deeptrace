from pathlib import Path
import tempfile

import pytest

from app.core.dense import DenseSettings, SentenceTransformerEmbedder
from app.evaluation.exact_dense import CachedExactDenseIndex
from app.models import Document


@pytest.mark.integration
def test_real_bge_exact_cache_and_reuse() -> None:
    settings = DenseSettings(model_name="BAAI/bge-m3", device="auto")
    embedder = SentenceTransformerEmbedder(settings)
    documents = [
        Document(
            id="d1",
            title="BM25",
            content="lexical keyword retrieval",
        ),
        Document(
            id="d2",
            title="Grounded answers",
            content="cite evidence and refuse unsupported answers",
        ),
    ]
    with tempfile.TemporaryDirectory(
        prefix="real-bge-cache-",
        dir=".pytest_cache",
    ) as cache_directory:
        first = CachedExactDenseIndex(
            documents,
            corpus_hash="integration-corpus",
            settings=settings,
            cache_root=Path(cache_directory),
            embedder_factory=lambda _: embedder,
        )
        second = CachedExactDenseIndex(
            documents,
            corpus_hash="integration-corpus",
            settings=settings,
            cache_root=Path(cache_directory),
            embedder_factory=lambda _: embedder,
        )
        response = second.search(
            "answer only when there is supporting evidence",
            top_k=2,
        )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.vector_dimension == 1024
    assert second.embedder.device in {"cpu", "cuda"}
    assert response.hits[0].document.id == "d2"
