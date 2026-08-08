from pathlib import Path

from app.corpus import load_jsonl
from app.core.bm25 import BM25Index


def test_demo_corpus_contains_thirty_unique_explainable_documents() -> None:
    documents = load_jsonl(Path("data/sample_documents.jsonl"))

    assert len(documents) == 30
    assert [document.id for document in documents] == [
        f"doc-{index:03d}" for index in range(1, 31)
    ]
    assert len({document.metadata["topic"] for document in documents}) >= 15
    assert all(document.title and document.content for document in documents)


def test_expanded_corpus_supports_new_exact_term_examples() -> None:
    index = BM25Index()
    index.build(load_jsonl(Path("data/sample_documents.jsonl")))

    assert index.search("HNSW efSearch").hits[0].document.id == "doc-013"
    assert (
        index.search("ANN Recall Retrieval Recall").hits[0].document.id
        == "doc-029"
    )
