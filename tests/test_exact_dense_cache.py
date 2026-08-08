from pathlib import Path
import tempfile

import numpy as np

from app.core.dense import DenseSettings
from app.evaluation.exact_dense import (
    CachedExactDenseIndex,
    embedding_cache_key,
)
from app.models import Document


class CountingEmbedder:
    model_name = "counting-model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if text == "semantic query":
                vectors.append([0.0, 1.0])
            elif "second" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def documents() -> list[Document]:
    return [
        Document(id="d1", title="first", content="first body"),
        Document(id="d2", title="second", content="second body"),
    ]


def test_cache_key_changes_with_model_or_corpus() -> None:
    first = embedding_cache_key(model_name="m1", corpus_hash="c1")
    assert first != embedding_cache_key(model_name="m2", corpus_hash="c1")
    assert first != embedding_cache_key(model_name="m1", corpus_hash="c2")


def test_document_vectors_are_cached_and_reused() -> None:
    embedder = CountingEmbedder()
    settings = DenseSettings(model_name=embedder.model_name)
    with tempfile.TemporaryDirectory(
        prefix="fake-dense-cache-",
        dir=".pytest_cache",
    ) as cache_directory:
        first = CachedExactDenseIndex(
            documents(),
            corpus_hash="corpus-hash",
            settings=settings,
            cache_root=Path(cache_directory),
            embedder_factory=lambda _: embedder,
        )
        assert first.cache_hit is False
        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == 2

        second = CachedExactDenseIndex(
            documents(),
            corpus_hash="corpus-hash",
            settings=settings,
            cache_root=Path(cache_directory),
            embedder_factory=lambda _: embedder,
        )
        assert second.cache_hit is True
        assert len(embedder.calls) == 1
        response = second.search("semantic query", top_k=1)
        second.search("semantic query", top_k=1)
        assert len(embedder.calls) == 2
        second.clear_query_cache()
        second.search("semantic query", top_k=1)

    assert response.hits[0].document.id == "d2"
    assert len(embedder.calls) == 3
    assert embedder.calls[-1] == ["semantic query"]
