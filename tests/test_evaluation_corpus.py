from __future__ import annotations

from app.evaluation.corpus import content_hash, normalize_text, stable_chunk_id


def test_normalized_hash_ignores_whitespace() -> None:
    assert normalize_text("  Hybrid\n SEARCH ") == "Hybrid SEARCH"
    assert content_hash("Hybrid SEARCH") == content_hash(" Hybrid  SEARCH ")


def test_chunk_ids_are_stable_and_namespaced() -> None:
    first = stable_chunk_id("scifact", "document-42")
    assert first == stable_chunk_id("scifact", "document-42")
    assert first != stable_chunk_id("trec-covid", "document-42")
