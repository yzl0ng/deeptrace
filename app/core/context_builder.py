from __future__ import annotations

from dataclasses import dataclass

from app.models import HybridSearchHit


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    hits: list[HybridSearchHit]
    characters: int
    truncated: bool


class ContextBuilder:
    """Turn ranked hybrid hits into deterministic, citation-safe LLM context."""

    def build(
        self,
        query: str,
        hybrid_hits: list[HybridSearchHit],
        max_chars: int,
    ) -> BuiltContext:
        del query  # Reserved for future query-aware context policies.
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")

        blocks: list[str] = []
        included_hits: list[HybridSearchHit] = []
        truncated = False
        used = 0

        for hit in hybrid_hits:
            header = self._header(hit)
            prefix = "" if not blocks else "\n\n"
            available = max_chars - used - len(prefix)
            if available < len(header):
                truncated = True
                break

            full_block = f"{header}{hit.document.content}"
            if len(full_block) <= available:
                block = full_block
            else:
                content_budget = available - len(header)
                block = f"{header}{hit.document.content[:content_budget]}"
                truncated = True

            blocks.append(f"{prefix}{block}")
            included_hits.append(hit)
            used += len(prefix) + len(block)
            if truncated:
                break

        if len(included_hits) < len(hybrid_hits):
            truncated = True
        text = "".join(blocks)
        return BuiltContext(
            text=text,
            hits=included_hits,
            characters=len(text),
            truncated=truncated,
        )

    @staticmethod
    def _header(hit: HybridSearchHit) -> str:
        source = hit.document.source or "unknown"
        metadata = hit.document.metadata
        location_lines: list[str] = []
        if metadata.get("document_id"):
            location_lines.append(
                f"Document ID: {metadata['document_id']}"
            )
        if metadata.get("chunk_id"):
            location_lines.append(f"Chunk ID: {metadata['chunk_id']}")
        if metadata.get("filename"):
            location_lines.append(f"Filename: {metadata['filename']}")
        if metadata.get("page_number") is not None:
            location_lines.append(
                f"Page: {metadata['page_number']}"
            )
        if metadata.get("section"):
            location_lines.append(f"Section: {metadata['section']}")
        if hit.rrf_rank is None:
            rank_lines = [f"RRF Rank: {hit.rank}"]
        else:
            rank_lines = [
                f"Rerank Rank: {hit.rank}",
                f"RRF Rank: {hit.rrf_rank}",
            ]
        if "bm25" in hit.source_ranks:
            rank_lines.append(f"BM25 Rank: {hit.source_ranks['bm25']}")
        if "dense" in hit.source_ranks:
            rank_lines.append(f"Dense Rank: {hit.source_ranks['dense']}")
        details = "\n".join([*location_lines, *rank_lines])
        return (
            f"[DOC {hit.document.id}]\n"
            f"Title: {hit.document.title}\n"
            f"Source: {source}\n"
            f"{details}\n"
            "Content:\n"
        )
