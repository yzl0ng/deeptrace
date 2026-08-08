from __future__ import annotations

import hashlib
import re
from typing import Sequence

from app.ingestion.models import ChunkDraft, ParsedBlock
from app.ingestion.parsers import normalize_text
from app.ingestion.security import validate_chunk_parameters


class DeterministicChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        validate_chunk_parameters(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        document_id: str,
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for block in blocks:
            for text in self._split_text(block.text):
                chunk_index = len(drafts)
                normalized = normalize_text(text)
                if not normalized:
                    continue
                content_hash = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()
                chunk_id = stable_chunk_id(
                    document_id,
                    chunk_index,
                    normalized,
                )
                drafts.append(
                    ChunkDraft(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        title=block.title,
                        section=block.section,
                        page_number=block.page_number,
                        text=normalized,
                        content_hash=content_hash,
                        metadata=dict(block.metadata),
                    )
                )
        return drafts

    def _split_text(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            hard_end = min(start + self.chunk_size, len(normalized))
            end = hard_end
            if hard_end < len(normalized):
                candidate = normalized[start:hard_end]
                boundary = _best_boundary(candidate)
                if boundary >= max(80, self.chunk_size // 2):
                    end = start + boundary
            piece = normalized[start:end].strip()
            if piece and (not chunks or piece != chunks[-1]):
                chunks.append(piece)
            if end >= len(normalized):
                break
            next_start = max(start + 1, end - self.chunk_overlap)
            next_start = _avoid_english_midword(normalized, next_start, end)
            if len(normalized) - next_start <= self.chunk_overlap:
                tail = normalized[next_start:].strip()
                if tail and tail not in chunks[-1]:
                    chunks.append(tail)
                break
            start = next_start
        return chunks


def stable_document_id(corpus_namespace: str, content_hash: str) -> str:
    digest = hashlib.sha256(
        f"{corpus_namespace}:{content_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return f"document-{digest}"


def stable_chunk_id(
    document_id: str,
    chunk_index: int,
    normalized_text: str,
) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{chunk_index}:{normalized_text}".encode("utf-8")
    ).hexdigest()[:24]
    return f"chunk-{digest}"


def _best_boundary(candidate: str) -> int:
    minimum = len(candidate) // 2
    patterns = (
        r"\n\n",
        r"\n",
        r"[。！？.!?]\s*",
        r"[；;]\s*",
        r"\s+",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, candidate[minimum:]))
        if matches:
            return minimum + matches[-1].end()
    return len(candidate)


def _avoid_english_midword(text: str, start: int, upper_bound: int) -> int:
    if start <= 0 or start >= len(text):
        return start
    if text[start - 1].isascii() and text[start - 1].isalnum():
        if text[start].isascii() and text[start].isalnum():
            cursor = start
            while cursor > 0 and cursor > start - 40:
                if text[cursor - 1].isspace():
                    return cursor
                cursor -= 1
    return min(start, upper_bound)
