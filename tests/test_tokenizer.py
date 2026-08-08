from app.core.tokenizer import normalize_text, tokenize


def test_normalize_full_width_and_case() -> None:
    assert normalize_text(" ＲＡＧ Search ") == "rag search"


def test_tokenize_mixed_technical_text() -> None:
    tokens = tokenize("BM25适合关键词检索")
    assert "bm25" in tokens
    assert "zh:关键" in tokens
    assert "zh:检索" in tokens


def test_tokenizer_is_deterministic() -> None:
    text = "Cross-Encoder 与 reranker"
    assert tokenize(text) == tokenize(text)
