from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    DeterministicChunker,
    stable_chunk_id,
    stable_document_id,
)
from app.ingestion.models import ParsedBlock


def test_empty_text_produces_no_chunks() -> None:
    chunks = DeterministicChunker().split(
        [ParsedBlock(title="Empty", text=" \n ")],
        document_id="document-empty",
    )
    assert chunks == []


def test_short_text_is_one_chunk_with_metadata() -> None:
    chunks = DeterministicChunker(200, 20).split(
        [
            ParsedBlock(
                title="Guide",
                section="Install",
                page_number=3,
                text="Install SearchLab with Conda.",
            )
        ],
        document_id="document-guide",
    )
    assert len(chunks) == 1
    assert chunks[0].section == "Install"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0


def test_long_paragraph_splits_with_bounded_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(180))
    chunks = DeterministicChunker(240, 40).split(
        [ParsedBlock(title="Long", text=text)],
        document_id="document-long",
    )
    assert len(chunks) > 2
    assert all(0 < len(chunk.text) <= 240 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)
    assert chunks[-1].text != chunks[-2].text


def test_chunking_is_deterministic_and_ids_are_stable() -> None:
    block = ParsedBlock(title="Stable", text=("确定性切分。" * 100))
    chunker = DeterministicChunker(220, 30)
    first = chunker.split([block], document_id="document-stable")
    second = chunker.split([block], document_id="document-stable")
    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_stable_ids_do_not_contain_local_paths() -> None:
    document_id = stable_document_id("uploaded", "abc123")
    chunk_id = stable_chunk_id(document_id, 0, "normalized text")
    assert document_id.startswith("document-")
    assert chunk_id.startswith("chunk-")
    assert "\\" not in document_id + chunk_id


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(199, 10), (4001, 10), (300, 300), (300, 301), (300, -1)],
)
def test_invalid_chunk_parameters_are_rejected(
    size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        DeterministicChunker(size, overlap)
