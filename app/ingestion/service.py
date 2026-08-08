from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4

from app.ingestion.chunker import DeterministicChunker, stable_document_id
from app.ingestion.models import PreparedIngestion
from app.ingestion.parsers import (
    DocumentParseError,
    normalized_document_content,
    parse_document,
)
from app.ingestion.security import (
    UploadSettings,
    UploadValidationError,
    validate_chunk_parameters,
    validate_upload,
)
from app.storage.repositories import ChunkWrite, DocumentRepository


class RebuildableIndex(Protocol):
    def rebuild(self, *, eager_dense: bool = False) -> int: ...


class IngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        index_manager: RebuildableIndex,
        settings: UploadSettings,
    ) -> None:
        self.repository = repository
        self.index_manager = index_manager
        self.settings = settings
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    def prepare_upload(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int,
        title: str | None = None,
        source: str | None = None,
    ) -> PreparedIngestion:
        validate_chunk_parameters(chunk_size, chunk_overlap)
        safe_name, file_type, mime_type = validate_upload(
            filename,
            content_type,
            content,
            max_bytes=self.settings.max_upload_bytes,
        )
        blocks = parse_document(
            content,
            file_type=file_type,
            filename=safe_name,
            title_override=title,
        )
        normalized_content = normalized_document_content(blocks)
        if not normalized_content:
            raise DocumentParseError(
                "no_extractable_text",
                "The file does not contain extractable text.",
            )
        content_hash = hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()
        document_id = stable_document_id("uploaded", content_hash)
        stored_filename = f"{document_id}.{file_type}"
        stored_path = self.settings.upload_dir / stored_filename

        existing = self.repository.get_document(document_id)
        duplicate = (
            existing is not None
            and existing.status == "completed"
            and existing.chunk_size == chunk_size
            and existing.chunk_overlap == chunk_overlap
        )
        if existing is None:
            _atomic_write(stored_path, content)
            document, _ = self.repository.create_document(
                document_id=document_id,
                corpus_namespace="uploaded",
                original_filename=safe_name,
                stored_filename=stored_filename,
                file_type=file_type,
                mime_type=mime_type,
                source=source,
                content_hash=content_hash,
                size_bytes=len(content),
                status="pending",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if document.document_id != document_id:
                raise RuntimeError("Unable to create the uploaded document.")
        job_id = f"ingestion-{uuid4().hex[:24]}"
        job = self.repository.create_job(
            job_id=job_id,
            document_id=document_id,
            status="completed" if duplicate else "pending",
            current_stage="duplicate" if duplicate else "pending",
            progress=100 if duplicate else 0,
        )
        if duplicate:
            self.repository.update_job(
                job.job_id,
                status="completed",
                current_stage="duplicate",
                progress=100,
            )
        else:
            self.repository.update_document_status(
                document_id,
                "pending",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return PreparedIngestion(
            document_id=document_id,
            job_id=job_id,
            original_filename=safe_name,
            stored_path=stored_path,
            file_type=file_type,
            mime_type=mime_type,
            content_hash=content_hash,
            title=(title or Path(safe_name).stem).strip(),
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            blocks=tuple(blocks),
            duplicate=duplicate,
        )

    def process(self, prepared: PreparedIngestion) -> None:
        if prepared.duplicate:
            return
        job_id = prepared.job_id
        document_id = prepared.document_id
        try:
            self._stage(job_id, document_id, "validating", 10)
            if not prepared.stored_path.is_file():
                raise UploadValidationError(
                    "stored_file_missing",
                    "The stored upload is unavailable.",
                )
            self._stage(job_id, document_id, "parsing", 25)
            blocks = list(prepared.blocks)
            self._stage(job_id, document_id, "chunking", 40)
            chunks = DeterministicChunker(
                prepared.chunk_size,
                prepared.chunk_overlap,
            ).split(blocks, document_id=document_id)
            if not chunks:
                raise DocumentParseError(
                    "no_chunks",
                    "The parsed document produced no non-empty chunks.",
                )
            self._stage(job_id, document_id, "storing", 55)
            self.repository.replace_chunks(
                document_id,
                [
                    ChunkWrite(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        corpus_namespace="uploaded",
                        chunk_index=chunk.chunk_index,
                        title=chunk.title,
                        section=chunk.section,
                        page_number=chunk.page_number,
                        text=chunk.text,
                        content_hash=chunk.content_hash,
                        token_count=chunk.token_count,
                        metadata=chunk.metadata,
                    )
                    for chunk in chunks
                ],
            )
            self._stage(job_id, document_id, "embedding", 70)
            self._stage(job_id, document_id, "indexing", 85)
            version = self.index_manager.rebuild(eager_dense=True)
            self.repository.update_document_status(
                document_id,
                "completed",
                index_version=version,
            )
            self.repository.update_job(
                job_id,
                status="completed",
                current_stage="completed",
                progress=100,
            )
        except Exception as error:
            code = getattr(error, "code", "ingestion_failed")
            message = _safe_error_message(error)
            self.repository.update_document_status(
                document_id,
                "failed",
                error_code=code,
                error_message=message,
            )
            self.repository.update_job(
                job_id,
                status="failed",
                current_stage="failed",
                progress=100,
                error_code=code,
                error_message=message,
            )

    def _stage(
        self,
        job_id: str,
        document_id: str,
        stage: str,
        progress: int,
    ) -> None:
        self.repository.update_document_status(document_id, stage)
        self.repository.update_job(
            job_id,
            status=stage,
            current_stage=stage,
            progress=progress,
        )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".upload-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip() or "Document ingestion failed."
    if ":\\" in message or ":/" in message:
        return "Document ingestion failed without changing the active index."
    return message[:500]
