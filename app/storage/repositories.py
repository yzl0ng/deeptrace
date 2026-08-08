from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import sqlite3
from typing import Any, Iterable, Sequence

import numpy as np

from app.models import Document
from app.storage.database import Database


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StoredDocument:
    document_id: str
    corpus_namespace: str
    original_filename: str
    stored_filename: str | None
    file_type: str
    mime_type: str
    source: str | None
    content_hash: str
    size_bytes: int
    status: str
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    created_at: str
    updated_at: str
    error_code: str | None
    error_message: str | None
    index_version: int


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: str
    document_id: str
    corpus_namespace: str
    chunk_index: int
    title: str
    section: str | None
    page_number: int | None
    text: str
    content_hash: str
    token_count: int | None
    created_at: str
    metadata: dict[str, Any]

    def to_search_document(self, original_filename: str) -> Document:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "chunk_index": self.chunk_index,
                "corpus_namespace": self.corpus_namespace,
                "filename": original_filename,
                "page_number": self.page_number,
                "section": self.section,
                "content_hash": self.content_hash,
            }
        )
        return Document(
            id=self.chunk_id,
            title=self.title,
            content=self.text,
            source=original_filename,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class StoredJob:
    job_id: str
    document_id: str
    status: str
    current_stage: str
    progress: int
    error_code: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class ChunkWrite:
    chunk_id: str
    document_id: str
    corpus_namespace: str
    chunk_index: int
    title: str
    section: str | None
    page_number: int | None
    text: str
    content_hash: str
    token_count: int | None
    metadata: dict[str, Any]


class ImmutableCorpusError(ValueError):
    pass


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def create_document(
        self,
        *,
        document_id: str,
        corpus_namespace: str,
        original_filename: str,
        stored_filename: str | None,
        file_type: str,
        mime_type: str,
        source: str | None,
        content_hash: str,
        size_bytes: int,
        status: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[StoredDocument, bool]:
        now = utc_now()
        with self.database.connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM documents
                WHERE corpus_namespace = ? AND content_hash = ?
                """,
                (corpus_namespace, content_hash),
            ).fetchone()
            if existing is not None:
                return _document_from_row(existing), False
            connection.execute(
                """
                INSERT INTO documents(
                    document_id, corpus_namespace, original_filename,
                    stored_filename, file_type, mime_type, source, content_hash,
                    size_bytes, status, chunk_size, chunk_overlap, chunk_count,
                    created_at, updated_at, index_version
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0)
                """,
                (
                    document_id,
                    corpus_namespace,
                    original_filename,
                    stored_filename,
                    file_type,
                    mime_type,
                    source,
                    content_hash,
                    size_bytes,
                    status,
                    chunk_size,
                    chunk_overlap,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return _document_from_row(row), True

    def seed_demo(self, documents: Sequence[Document]) -> int:
        inserted = 0
        now = utc_now()
        with self.database.connect() as connection:
            for document in documents:
                content_hash = _sha256_text(
                    f"{document.title}\n{document.content}"
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO documents(
                        document_id, corpus_namespace, original_filename,
                        stored_filename, file_type, mime_type, source,
                        content_hash, size_bytes, status, chunk_size,
                        chunk_overlap, chunk_count, created_at, updated_at,
                        index_version
                    )
                    VALUES(?, 'demo', 'sample_documents.jsonl', NULL, 'jsonl',
                           'application/x-ndjson', ?, ?, ?, 'completed',
                           800, 120, 1, ?, ?, 0)
                    """,
                    (
                        document.id,
                        document.source,
                        content_hash,
                        len(document.content.encode("utf-8")),
                        now,
                        now,
                    ),
                )
                if cursor.rowcount:
                    inserted += 1
                connection.execute(
                    """
                    INSERT OR IGNORE INTO chunks(
                        chunk_id, document_id, corpus_namespace, chunk_index,
                        title, section, page_number, text, content_hash,
                        token_count, created_at, metadata_json
                    )
                    VALUES(?, ?, 'demo', 0, ?, NULL, NULL, ?, ?, NULL, ?, ?)
                    """,
                    (
                        document.id,
                        document.id,
                        document.title,
                        document.content,
                        _sha256_text(document.content),
                        now,
                        json.dumps(document.metadata, ensure_ascii=False),
                    ),
                )
        return inserted

    def get_document(self, document_id: str) -> StoredDocument | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return _document_from_row(row) if row is not None else None

    def list_documents(
        self,
        *,
        corpus_namespace: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StoredDocument], int]:
        clauses: list[str] = []
        parameters: list[object] = []
        if corpus_namespace:
            clauses.append("corpus_namespace = ?")
            parameters.append(corpus_namespace)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT count(*) FROM documents {where}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM documents
                {where}
                ORDER BY created_at DESC, document_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
        return [_document_from_row(row) for row in rows], total

    def list_chunks(
        self,
        document_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[StoredChunk], int]:
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    "SELECT count(*) FROM chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                LIMIT ? OFFSET ?
                """,
                (document_id, limit, offset),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows], total

    def list_index_chunks(self) -> list[tuple[StoredChunk, str]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.original_filename
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE d.corpus_namespace IN ('demo', 'uploaded')
                  AND d.status IN ('completed', 'indexing')
                ORDER BY d.corpus_namespace, d.document_id, c.chunk_index
                """
            ).fetchall()
        return [
            (_chunk_from_row(row), str(row["original_filename"]))
            for row in rows
        ]

    def replace_chunks(
        self,
        document_id: str,
        chunks: Sequence[ChunkWrite],
    ) -> None:
        document = self.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        if document.corpus_namespace == "evaluation":
            raise ImmutableCorpusError("Evaluation corpus is immutable.")
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, document_id, corpus_namespace, chunk_index,
                    title, section, page_number, text, content_hash,
                    token_count, created_at, metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.corpus_namespace,
                        chunk.chunk_index,
                        chunk.title,
                        chunk.section,
                        chunk.page_number,
                        chunk.text,
                        chunk.content_hash,
                        chunk.token_count,
                        now,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                """
                UPDATE documents
                SET chunk_count = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (len(chunks), now, document_id),
            )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        index_version: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        assignments = [
            "status = ?",
            "error_code = ?",
            "error_message = ?",
            "updated_at = ?",
        ]
        parameters: list[object] = [
            status,
            error_code,
            error_message,
            utc_now(),
        ]
        if index_version is not None:
            assignments.append("index_version = ?")
            parameters.append(index_version)
        if chunk_size is not None:
            assignments.append("chunk_size = ?")
            parameters.append(chunk_size)
        if chunk_overlap is not None:
            assignments.append("chunk_overlap = ?")
            parameters.append(chunk_overlap)
        parameters.append(document_id)
        with self.database.connect() as connection:
            connection.execute(
                f"""
                UPDATE documents SET {', '.join(assignments)}
                WHERE document_id = ?
                """,
                parameters,
            )

    def create_job(
        self,
        *,
        job_id: str,
        document_id: str,
        status: str = "pending",
        current_stage: str = "pending",
        progress: int = 0,
    ) -> StoredJob:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs(
                    job_id, document_id, status, current_stage, progress,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    document_id,
                    status,
                    current_stage,
                    progress,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(row)

    def get_job(self, job_id: str) -> StoredJob | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        current_stage: str,
        progress: int,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = utc_now()
        with self.database.connect() as connection:
            current = connection.execute(
                "SELECT started_at FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if current is None:
                raise KeyError(job_id)
            started_at = current["started_at"]
            if started_at is None and status not in {"pending"}:
                started_at = now
            completed_at = now if status in {"completed", "failed"} else None
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, current_stage = ?, progress = ?,
                    error_code = ?, error_message = ?, started_at = ?,
                    completed_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    current_stage,
                    progress,
                    error_code,
                    error_message,
                    started_at,
                    completed_at,
                    job_id,
                ),
            )

    def get_embeddings(
        self,
        chunks: Sequence[StoredChunk],
        *,
        model_name: str,
        normalized: bool,
    ) -> dict[str, np.ndarray]:
        if not chunks:
            return {}
        placeholders = ",".join("?" for _ in chunks)
        parameters: list[object] = [
            model_name,
            int(normalized),
            *(chunk.chunk_id for chunk in chunks),
        ]
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT e.*
                FROM embeddings e
                WHERE e.model_name = ?
                  AND e.normalized = ?
                  AND e.chunk_id IN ({placeholders})
                """,
                parameters,
            ).fetchall()
        hashes = {chunk.chunk_id: chunk.content_hash for chunk in chunks}
        vectors: dict[str, np.ndarray] = {}
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if str(row["content_hash"]) != hashes.get(chunk_id):
                continue
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32).copy()
            if len(vector) != int(row["dimension"]):
                continue
            vectors[chunk_id] = vector
        return vectors

    def upsert_embeddings(
        self,
        records: Iterable[
            tuple[str, str, bool, str, np.ndarray]
        ],
    ) -> None:
        now = utc_now()
        rows = []
        for chunk_id, model_name, normalized, content_hash, vector in records:
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    chunk_id,
                    model_name,
                    len(array),
                    int(normalized),
                    content_hash,
                    array.tobytes(),
                    now,
                )
            )
        if not rows:
            return
        with self.database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO embeddings(
                    chunk_id, model_name, dimension, normalized,
                    content_hash, vector_blob, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, model_name, normalized) DO UPDATE SET
                    dimension = excluded.dimension,
                    content_hash = excluded.content_hash,
                    vector_blob = excluded.vector_blob,
                    created_at = excluded.created_at
                """,
                rows,
            )

    def delete_document(self, document_id: str) -> StoredDocument:
        document = self.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        if document.corpus_namespace in {"demo", "evaluation"}:
            raise ImmutableCorpusError(
                f"{document.corpus_namespace} corpus is read-only."
            )
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
        return document

    def counts(self) -> dict[str, int]:
        with self.database.connect() as connection:
            documents = int(
                connection.execute(
                    """
                    SELECT count(*) FROM documents
                    WHERE corpus_namespace IN ('demo', 'uploaded')
                      AND status = 'completed'
                    """
                ).fetchone()[0]
            )
            chunks = int(
                connection.execute(
                    """
                    SELECT count(*) FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE d.corpus_namespace IN ('demo', 'uploaded')
                      AND d.status = 'completed'
                    """
                ).fetchone()[0]
            )
        return {"documents": documents, "chunks": chunks}

    def current_index_version(self) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute(
                    """
                    SELECT current_version FROM index_state
                    WHERE singleton = 1
                    """
                ).fetchone()[0]
            )

    def set_index_version(self, version: int) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE index_state
                SET current_version = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (version, utc_now()),
            )


def _document_from_row(row: sqlite3.Row) -> StoredDocument:
    return StoredDocument(
        **{field: row[field] for field in StoredDocument.__dataclass_fields__}
    )


def _chunk_from_row(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        corpus_namespace=str(row["corpus_namespace"]),
        chunk_index=int(row["chunk_index"]),
        title=str(row["title"]),
        section=row["section"],
        page_number=row["page_number"],
        text=str(row["text"]),
        content_hash=str(row["content_hash"]),
        token_count=row["token_count"],
        created_at=str(row["created_at"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _job_from_row(row: sqlite3.Row) -> StoredJob:
    return StoredJob(
        **{field: row[field] for field in StoredJob.__dataclass_fields__}
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
