from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ALLOWED_TYPES = {
    ".txt": {
        "text/plain",
        "application/octet-stream",
    },
    ".md": {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    },
    ".pdf": {
        "application/pdf",
        "application/octet-stream",
    },
}


class UploadValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class UploadSettings:
    upload_dir: Path
    max_upload_mb: int = 20
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120

    @classmethod
    def from_environment(cls, project_root: Path) -> UploadSettings:
        upload_dir = Path(
            os.getenv("SEARCHLAB_UPLOAD_DIR", "data/uploads")
        )
        if not upload_dir.is_absolute():
            upload_dir = project_root / upload_dir
        return cls(
            upload_dir=upload_dir,
            max_upload_mb=int(os.getenv("SEARCHLAB_MAX_UPLOAD_MB", "20")),
            default_chunk_size=int(
                os.getenv("SEARCHLAB_DEFAULT_CHUNK_SIZE", "800")
            ),
            default_chunk_overlap=int(
                os.getenv("SEARCHLAB_DEFAULT_CHUNK_OVERLAP", "120")
            ),
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def validate_upload(
    filename: str | None,
    content_type: str | None,
    content: bytes,
    *,
    max_bytes: int,
) -> tuple[str, str, str]:
    if not filename:
        raise UploadValidationError(
            "missing_filename",
            "The uploaded file must have a filename.",
        )
    safe_name = Path(filename.replace("\\", "/")).name
    if safe_name in {"", ".", ".."}:
        raise UploadValidationError(
            "invalid_filename",
            "The uploaded filename is invalid.",
        )
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_TYPES:
        raise UploadValidationError(
            "unsupported_file_type",
            "Only PDF, Markdown and TXT files are supported.",
        )
    normalized_mime = (content_type or "application/octet-stream").lower()
    normalized_mime = normalized_mime.split(";", 1)[0].strip()
    if normalized_mime not in ALLOWED_TYPES[extension]:
        raise UploadValidationError(
            "mime_type_mismatch",
            "The file extension does not match its MIME type.",
        )
    if not content:
        raise UploadValidationError(
            "empty_file",
            "The uploaded file is empty.",
        )
    if len(content) > max_bytes:
        raise UploadValidationError(
            "file_too_large",
            "The uploaded file exceeds the configured size limit.",
        )
    return safe_name, extension.removeprefix("."), normalized_mime


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if not 200 <= chunk_size <= 4000:
        raise UploadValidationError(
            "invalid_chunk_size",
            "chunk_size must be between 200 and 4000 characters.",
        )
    if not 0 <= chunk_overlap <= 1000:
        raise UploadValidationError(
            "invalid_chunk_overlap",
            "chunk_overlap must be between 0 and 1000 characters.",
        )
    if chunk_overlap >= chunk_size:
        raise UploadValidationError(
            "invalid_chunk_overlap",
            "chunk_overlap must be smaller than chunk_size.",
        )
