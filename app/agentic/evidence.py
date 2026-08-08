from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.bm25 import BM25Index
from app.core.hybrid import HybridRetriever
from app.models import Document


def utc_now() -> datetime:
    return datetime.now(UTC)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    canonical_url: str
    discovered_url: str
    title: str
    snippet: str | None = None
    provider: str
    search_rank: int | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class WebDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str
    canonical_url: str
    title: str
    content: str
    content_hash: str
    media_type: str
    fetched_at: datetime
    expires_at: datetime
    status: str = "ready"


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    document_id: str
    source_id: str
    ordinal: int
    content: str
    start_char: int
    end_char: int
    content_hash: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    query: str
    passage_id: str
    document_id: str
    source_id: str
    canonical_url: str
    title: str
    content: str
    rank: int
    retrieval_method: str
    score: float
    trace: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EvidenceStore:
    """Durable, deduplicated provenance store for untrusted web content."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def upsert_source(
        self,
        *,
        canonical_url: str,
        discovered_url: str,
        title: str,
        snippet: str | None,
        provider: str,
        search_rank: int | None,
    ) -> Source:
        source_id = _stable_id("src", canonical_url)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    source_id, canonical_url, discovered_url, title, snippet,
                    provider, search_rank, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                    discovered_url = excluded.discovered_url,
                    title = excluded.title,
                    snippet = excluded.snippet,
                    provider = excluded.provider,
                    search_rank = excluded.search_rank,
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    canonical_url,
                    discovered_url,
                    title,
                    snippet,
                    provider,
                    search_rank,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> Source:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        if row is None:
            raise KeyError(source_id)
        return Source.model_validate(dict(row))

    def get_fresh_document(
        self,
        canonical_url: str,
        *,
        now: datetime | None = None,
    ) -> WebDocument | None:
        checked_at = now or utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE canonical_url = ? AND status = 'ready'
                  AND expires_at > ?
                ORDER BY fetched_at DESC LIMIT 1
                """,
                (canonical_url, checked_at.isoformat()),
            ).fetchone()
        return WebDocument.model_validate(dict(row)) if row else None

    def store_document(
        self,
        *,
        source: Source,
        title: str,
        content: str,
        media_type: str,
        fetched_at: datetime,
        cache_ttl: timedelta,
        chunk_size: int = 1200,
        chunk_overlap: int = 160,
    ) -> tuple[WebDocument, list[Passage], bool]:
        normalized_content = _normalize_text(content)
        content_hash = hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()
        document_id = _stable_id(
            "doc", f"{source.canonical_url}\0{content_hash}"
        )
        existing = self._get_document(document_id)
        if existing is not None:
            refreshed = existing.model_copy(
                update={
                    "fetched_at": fetched_at,
                    "expires_at": fetched_at + cache_ttl,
                }
            )
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE documents
                    SET fetched_at = ?, expires_at = ?
                    WHERE document_id = ?
                    """,
                    (
                        refreshed.fetched_at.isoformat(),
                        refreshed.expires_at.isoformat(),
                        document_id,
                    ),
                )
            return (
                refreshed,
                self.passages_for_document(document_id),
                False,
            )

        document = WebDocument(
            document_id=document_id,
            source_id=source.source_id,
            canonical_url=source.canonical_url,
            title=title or source.title,
            content=normalized_content,
            content_hash=content_hash,
            media_type=media_type,
            fetched_at=fetched_at,
            expires_at=fetched_at + cache_ttl,
        )
        passages = _chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_id, canonical_url, title, content,
                    content_hash, media_type, fetched_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.source_id,
                    document.canonical_url,
                    document.title,
                    document.content,
                    document.content_hash,
                    document.media_type,
                    document.fetched_at.isoformat(),
                    document.expires_at.isoformat(),
                    document.status,
                ),
            )
            connection.executemany(
                """
                INSERT INTO passages (
                    passage_id, document_id, source_id, ordinal, content,
                    start_char, end_char, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.passage_id,
                        item.document_id,
                        item.source_id,
                        item.ordinal,
                        item.content,
                        item.start_char,
                        item.end_char,
                        item.content_hash,
                    )
                    for item in passages
                ],
            )
        return document, passages, True

    def passages_for_document(self, document_id: str) -> list[Passage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM passages
                WHERE document_id = ? ORDER BY ordinal
                """,
                (document_id,),
            ).fetchall()
        return [Passage.model_validate(dict(row)) for row in rows]

    def all_passages(self) -> list[Passage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM passages ORDER BY passage_id"
            ).fetchall()
        return [Passage.model_validate(dict(row)) for row in rows]

    def source_for_passage(self, passage_id: str) -> Source:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sources.* FROM sources
                JOIN passages ON passages.source_id = sources.source_id
                WHERE passages.passage_id = ?
                """,
                (passage_id,),
            ).fetchone()
        if row is None:
            raise KeyError(passage_id)
        return Source.model_validate(dict(row))

    def save_evidence(self, evidence: Iterable[Evidence]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO evidence (
                    evidence_id, query, passage_id, document_id, source_id,
                    canonical_url, title, content, rank, retrieval_method,
                    score, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.evidence_id,
                        item.query,
                        item.passage_id,
                        item.document_id,
                        item.source_id,
                        item.canonical_url,
                        item.title,
                        item.content,
                        item.rank,
                        item.retrieval_method,
                        item.score,
                        json.dumps(item.trace, ensure_ascii=False),
                        item.created_at.isoformat(),
                    )
                    for item in evidence
                ],
            )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                )
                for table in ("sources", "documents", "passages", "evidence")
            }

    def _get_document(self, document_id: str) -> WebDocument | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return WebDocument.model_validate(dict(row)) if row else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    canonical_url TEXT NOT NULL UNIQUE,
                    discovered_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    snippet TEXT,
                    provider TEXT NOT NULL,
                    search_rank INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(source_id),
                    canonical_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    UNIQUE(canonical_url, content_hash)
                );
                CREATE TABLE IF NOT EXISTS passages (
                    passage_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    source_id TEXT NOT NULL REFERENCES sources(source_id),
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    passage_id TEXT NOT NULL REFERENCES passages(passage_id),
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    source_id TEXT NOT NULL REFERENCES sources(source_id),
                    canonical_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    retrieval_method TEXT NOT NULL,
                    score REAL NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_url
                    ON documents(canonical_url);
                CREATE INDEX IF NOT EXISTS idx_passages_source
                    ON passages(source_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_query
                    ON evidence(query);
                """
            )


class EvidenceRetriever:
    """Reuse SearchLab BM25 and optionally its Dense/RRF/Reranker stack."""

    def __init__(
        self,
        store: EvidenceStore,
        *,
        dense_retriever: Any | None = None,
        reranker: Any | None = None,
        candidate_k: int = 20,
        rank_constant: int = 60,
    ) -> None:
        self.store = store
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant

    def search(self, query: str, *, top_k: int = 5) -> list[Evidence]:
        passages = self.store.all_passages()
        documents = [_passage_document(item, self.store) for item in passages]
        bm25 = BM25Index()
        bm25.build(documents)

        if self.dense_retriever is None:
            response = bm25.search(query, top_k=top_k)
            ranked = [
                (
                    hit.document,
                    float(hit.score),
                    {
                        "bm25_rank": hit.rank,
                        "bm25_score": hit.score,
                        "matched_terms": hit.matched_terms,
                    },
                )
                for hit in response.hits
            ]
            method = "bm25"
        else:
            build = getattr(self.dense_retriever, "build", None)
            if callable(build):
                build(documents)
            hybrid = HybridRetriever(
                bm25,
                self.dense_retriever,
                rank_constant=self.rank_constant,
                candidate_k=self.candidate_k,
            ).search(query, top_k=max(top_k, self.candidate_k))
            hits = hybrid.hits
            method = "rrf"
            if self.reranker is not None:
                hits = self.reranker.rerank(query, hits, top_k).hits
                method = "rrf_reranker"
            else:
                hits = hits[:top_k]
            ranked = [
                (
                    hit.document,
                    float(hit.reranker_score or hit.rrf_score),
                    {
                        "rrf_rank": hit.rrf_rank or hit.rank,
                        "rerank_rank": (
                            hit.rank if hit.reranker_score is not None else None
                        ),
                        "source_ranks": hit.source_ranks,
                        "source_scores": hit.source_scores,
                    },
                )
                for hit in hits
            ]

        evidence = [
            _to_evidence(query, rank, document, score, method, trace)
            for rank, (document, score, trace) in enumerate(ranked, start=1)
        ]
        self.store.save_evidence(evidence)
        return evidence


def _passage_document(passage: Passage, store: EvidenceStore) -> Document:
    source = store.source_for_passage(passage.passage_id)
    return Document(
        id=passage.passage_id,
        title=source.title,
        content=passage.content,
        source=source.canonical_url,
        metadata={
            "document_id": passage.document_id,
            "source_id": passage.source_id,
            "canonical_url": source.canonical_url,
        },
    )


def _to_evidence(
    query: str,
    rank: int,
    document: Document,
    score: float,
    method: str,
    trace: dict[str, Any],
) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{uuid4().hex}",
        query=query,
        passage_id=document.id,
        document_id=str(document.metadata["document_id"]),
        source_id=str(document.metadata["source_id"]),
        canonical_url=str(document.metadata["canonical_url"]),
        title=document.title,
        content=document.content,
        rank=rank,
        retrieval_method=method,
        score=score,
        trace=trace,
    )


def _chunk_document(
    document: WebDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Passage]:
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk size must exceed a non-negative overlap")
    passages: list[Passage] = []
    start = 0
    ordinal = 0
    while start < len(document.content):
        end = min(start + chunk_size, len(document.content))
        if end < len(document.content):
            boundary = document.content.rfind("\n", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        content = document.content[start:end].strip()
        if content:
            passages.append(
                Passage(
                    passage_id=_stable_id(
                        "psg", f"{document.document_id}\0{ordinal}"
                    ),
                    document_id=document.document_id,
                    source_id=document.source_id,
                    ordinal=ordinal,
                    content=content,
                    start_char=start,
                    end_char=end,
                    content_hash=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                )
            )
            ordinal += 1
        if end >= len(document.content):
            break
        start = end - chunk_overlap
    return passages


def _normalize_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", value.replace("\r\n", "\n")).strip()


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"
