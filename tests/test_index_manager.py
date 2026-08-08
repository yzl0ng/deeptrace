from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from app.core.dense import DenseSettings
from app.core.index_manager import IndexManager
from app.storage.database import Database
from app.storage.repositories import ChunkWrite, DocumentRepository


class ManagerFakeEmbedder:
    model_name = "fake-manager"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [
                [1.0, 0.0]
                if "ZephyrGraph" in text
                else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def make_repository(tmp_path: Path) -> DocumentRepository:
    repository = DocumentRepository(Database(tmp_path / "manager.db"))
    repository.create_document(
        document_id="document-z",
        corpus_namespace="uploaded",
        original_filename="z.txt",
        stored_filename="document-z.txt",
        file_type="txt",
        mime_type="text/plain",
        source=None,
        content_hash="z-file",
        size_bytes=20,
        status="indexing",
        chunk_size=800,
        chunk_overlap=120,
    )
    repository.replace_chunks(
        "document-z",
        [
            ChunkWrite(
                chunk_id="chunk-z",
                document_id="document-z",
                corpus_namespace="uploaded",
                chunk_index=0,
                title="Upload",
                section=None,
                page_number=None,
                text="ZephyrGraph unique retrieval term",
                content_hash="z-chunk",
                token_count=None,
                metadata={},
            )
        ],
    )
    return repository


def test_rebuild_makes_new_chunk_searchable_and_reuses_cache(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    embedder = ManagerFakeEmbedder()
    manager = IndexManager(
        repository,
        DenseSettings(model_name="fake-manager"),
        embedder_factory=lambda _: embedder,
    )
    first_version = manager.rebuild(eager_dense=True)
    snapshot = manager.current()

    assert snapshot.bm25.search("ZephyrGraph", 1).hits[0].document.id == "chunk-z"
    assert snapshot.dense.search("ZephyrGraph", 1).hits[0].document.id == "chunk-z"
    assert snapshot.hybrid.search(
        "ZephyrGraph", 1, candidate_k=1
    ).hits[0].document.id == "chunk-z"
    assert snapshot.dense.cache_misses == 1

    second_version = manager.rebuild(eager_dense=True)
    assert second_version == first_version + 1
    assert manager.current().dense.cache_hits == 1
    assert manager.current().dense.cache_misses == 0
    document_batches = [call for call in embedder.calls if "\n" in call[0]]
    assert len(document_batches) == 1


def test_failed_rebuild_preserves_old_snapshot(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    working = ManagerFakeEmbedder()
    manager = IndexManager(
        repository,
        DenseSettings(model_name="fake-manager"),
        embedder_factory=lambda _: working,
    )
    manager.rebuild(eager_dense=True)
    old_snapshot = manager.current()
    manager._embedder = None

    def fail(_: DenseSettings):
        raise RuntimeError("embedding failed")

    manager._embedder_factory = fail
    with pytest.raises(Exception, match="embedding failed"):
        manager.rebuild(eager_dense=True)
    assert manager.current() is old_snapshot
    assert manager.status()["last_error"] == "embedding failed"
