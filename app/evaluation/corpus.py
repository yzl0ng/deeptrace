from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field


WHITESPACE = re.compile(r"\s+")


class CorpusChunk(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    text: str
    source: str
    source_type: str
    language: str
    topic: str
    section: str
    page_number: int | None = None
    license: str
    corpus_split: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return WHITESPACE.sub(" ", normalized).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def stable_chunk_id(dataset: str, original_id: str) -> str:
    identity = f"{dataset}\0{original_id}".encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:20]
    safe_dataset = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
    return f"{safe_dataset}-chunk-{suffix}"
