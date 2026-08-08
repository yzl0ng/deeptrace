from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.models import Document
from app.storage.database import Database
from app.storage.repositories import (
    ChunkWrite,
    DocumentRepository,
    ImmutableCorpusError,
)


def repository_at(path: Path) -> DocumentRepository:
    return DocumentRepository(Database(path / "searchlab-test.db"))


def add_uploaded(repository: DocumentRepository) -> str:
    document_id = "document-uploaded"
    repository.create_document(
        document_id=document_id,
        corpus_namespace="uploaded",
        original_filename="upload.txt",
        stored_filename="document-uploaded.txt",
        file_type="txt",
        mime_type="text/plain",
        source=None,
        content_hash="file-hash",
        size_bytes=12,
        status="pending",
        chunk_size=800,
        chunk_overlap=120,
    )
    repository.replace_chunks(
        document_id,
        [
            ChunkWrite(
                chunk_id="chunk-uploaded",
                document_id=document_id,
                corpus_namespace="uploaded",
                chunk_index=0,
                title="Upload",
                section=None,
                page_number=None,
                text="ZephyrGraph",
                content_hash="chunk-hash",
                token_count=None,
                metadata={},
            )
        ],
    )
    repository.update_document_status(document_id, "completed")
    return document_id


def test_document_chunk_embedding_crud_and_cascade(tmp_path: Path) -> None:
    repository = repository_at(tmp_path)
    document_id = add_uploaded(repository)
    document = repository.get_document(document_id)
    chunks, total = repository.list_chunks(document_id)
    assert document is not None
    assert total == 1

    repository.upsert_embeddings(
        [
            (
                chunks[0].chunk_id,
                "fake",
                True,
                chunks[0].content_hash,
                np.asarray([1.0, 0.0], dtype=np.float32),
            )
        ]
    )
    cached = repository.get_embeddings(
        chunks,
        model_name="fake",
        normalized=True,
    )
    assert cached["chunk-uploaded"].tolist() == [1.0, 0.0]

    repository.delete_document(document_id)
    assert repository.get_document(document_id) is None
    assert repository.list_chunks(document_id)[1] == 0


def test_duplicate_content_hash_returns_existing_document(
    tmp_path: Path,
) -> None:
    repository = repository_at(tmp_path)
    document_id = add_uploaded(repository)
    duplicate, created = repository.create_document(
        document_id="different-id",
        corpus_namespace="uploaded",
        original_filename="renamed.txt",
        stored_filename="different-id.txt",
        file_type="txt",
        mime_type="text/plain",
        source=None,
        content_hash="file-hash",
        size_bytes=12,
        status="pending",
        chunk_size=800,
        chunk_overlap=120,
    )
    assert created is False
    assert duplicate.document_id == document_id


def test_demo_seed_is_idempotent_and_read_only(tmp_path: Path) -> None:
    repository = repository_at(tmp_path)
    demo = [Document(id="doc-001", title="BM25", content="keyword")]
    assert repository.seed_demo(demo) == 1
    assert repository.seed_demo(demo) == 0
    assert repository.list_documents(corpus_namespace="demo")[1] == 1
    with pytest.raises(ImmutableCorpusError):
        repository.delete_document("doc-001")


def test_document_list_paginates(tmp_path: Path) -> None:
    repository = repository_at(tmp_path)
    repository.seed_demo(
        [
            Document(id="doc-001", title="One", content="first"),
            Document(id="doc-002", title="Two", content="second"),
        ]
    )
    rows, total = repository.list_documents(limit=1, offset=1)
    assert total == 2
    assert len(rows) == 1


def test_job_status_updates_are_persisted(tmp_path: Path) -> None:
    repository = repository_at(tmp_path)
    document_id = add_uploaded(repository)
    repository.create_job(job_id="job-1", document_id=document_id)
    repository.update_job(
        "job-1",
        status="completed",
        current_stage="completed",
        progress=100,
    )
    job = repository.get_job("job-1")
    assert job is not None
    assert job.completed_at is not None
    assert job.progress == 100
