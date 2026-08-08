from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CorpusNamespace(StrEnum):
    DEMO = "demo"
    UPLOADED = "uploaded"
    EVALUATION = "evaluation"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    STORING = "storing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"
    REINDEXING = "reindexing"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    title: str
    text: str
    section: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    text: str
    content_hash: str
    section: str | None = None
    page_number: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedIngestion:
    document_id: str
    job_id: str
    original_filename: str
    stored_path: Path
    file_type: str
    mime_type: str
    content_hash: str
    title: str
    source: str | None
    chunk_size: int
    chunk_overlap: int
    blocks: tuple[ParsedBlock, ...]
    duplicate: bool = False


class DocumentSummary(BaseModel):
    document_id: str
    corpus_namespace: str
    original_filename: str
    file_type: str
    mime_type: str
    source: str | None
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


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentSummary]


class ChunkSummary(BaseModel):
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


class ChunkListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    chunks: list[ChunkSummary]


class IngestionAccepted(BaseModel):
    document_id: str
    job_id: str
    status: str
    duplicate: bool = False


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    current_stage: str
    progress: int = Field(ge=0, le=100)
    error_code: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: str
    index_version: int


class ReindexRequest(BaseModel):
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class IndexStatusResponse(BaseModel):
    current_version: int
    documents: int
    chunks: int
    bm25_ready: bool
    dense_ready: bool
    reranker_ready: bool
    rebuild_in_progress: bool
    last_build_time_ms: float
    last_error: str | None
    embedding_model: str
    embedding_device: str
    embedding_dimension: int


class RebuildResponse(BaseModel):
    status: str
    index_version: int
