from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from app.core.dense import (
    DenseIndex,
    DenseRuntime,
    DenseSettings,
    DenseUnavailableError,
    select_device,
)
from app.models import Document


class FakeEmbedder:
    model_name = "fake-embedding"
    device = "cpu"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def sample_documents() -> list[Document]:
    return [
        Document(id="a", title="Alpha", content="first"),
        Document(id="b", title="Beta", content="second"),
        Document(id="c", title="Gamma", content="third"),
    ]


def test_fixed_vectors_are_ranked_by_cosine_similarity() -> None:
    embedder = FakeEmbedder(
        {
            "Alpha\nfirst": [1.0, 0.0],
            "Beta\nsecond": [0.0, 1.0],
            "Gamma\nthird": [-1.0, 0.0],
            "semantic query": [0.9, 0.1],
        }
    )
    index = DenseIndex(embedder)
    index.build(sample_documents())

    response = index.search("semantic query", top_k=3)

    assert [hit.document.id for hit in response.hits] == ["a", "b", "c"]
    assert response.hits[0].score == pytest.approx(0.9938837)
    assert response.vector_dimension == 2


def test_document_vectors_are_built_once_and_query_only_embeds_query() -> None:
    embedder = FakeEmbedder(
        {
            "Alpha\nfirst": [1.0, 0.0],
            "Beta\nsecond": [0.0, 1.0],
            "Gamma\nthird": [-1.0, 0.0],
            "query one": [1.0, 0.0],
            "query two": [0.0, 1.0],
        }
    )
    index = DenseIndex(embedder)
    index.build(sample_documents())
    index.search("query one", top_k=2)
    index.search("query two", top_k=2)

    assert embedder.calls == [
        ["Alpha\nfirst", "Beta\nsecond", "Gamma\nthird"],
        ["query one"],
        ["query two"],
    ]


def test_empty_corpus_returns_empty_results_without_embedding() -> None:
    embedder = FakeEmbedder({})
    index = DenseIndex(embedder)
    index.build([])

    response = index.search("anything", top_k=5)

    assert response.hits == []
    assert response.total_hits == 0
    assert response.vector_dimension == 0
    assert embedder.calls == []


def test_top_k_limits_results() -> None:
    embedder = FakeEmbedder(
        {
            "Alpha\nfirst": [1.0, 0.0],
            "Beta\nsecond": [0.0, 1.0],
            "Gamma\nthird": [-1.0, 0.0],
            "query": [1.0, 0.0],
        }
    )
    index = DenseIndex(embedder)
    index.build(sample_documents())

    assert len(index.search("query", top_k=2).hits) == 2


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_device_selection(
    requested: str,
    cuda_available: bool,
    expected: str,
) -> None:
    assert (
        select_device(requested, cuda_available=cuda_available)
        == expected
    )


def test_explicit_cuda_fails_clearly_when_unavailable() -> None:
    with pytest.raises(DenseUnavailableError, match="CUDA is not available"):
        select_device("cuda", cuda_available=False)


def test_runtime_loads_model_and_builds_index_only_once() -> None:
    factory_calls = 0
    embedder = FakeEmbedder(
        {
            "Alpha\nfirst": [1.0, 0.0],
            "query one": [1.0, 0.0],
            "query two": [1.0, 0.0],
        }
    )

    def factory(_: DenseSettings) -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return embedder

    runtime = DenseRuntime(
        [sample_documents()[0]],
        DenseSettings(model_name="fake"),
        factory,
    )
    runtime.search("query one", 1)
    runtime.search("query two", 1)

    assert factory_calls == 1
    assert embedder.calls[0] == ["Alpha\nfirst"]
    assert runtime.stats().status == "ready"
