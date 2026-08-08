from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = (
    PROJECT_ROOT / "data" / "experiments" / "scifact_en" / "quality"
)
CORPUS_ROOT = PROJECT_ROOT / "data" / "corpora"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "evaluation"
REVIEWED_QUERY_COUNT = 38

PROPOSED_UNREVIEWED = [
    ("exact-term", "What does BM25 parameter b normalize?"),
    ("exact-term", "How does HNSW efSearch affect query latency?"),
    ("acronym", "What is the difference between ANN and exact KNN?"),
    ("acronym", "How does RRF merge ranked result lists?"),
    (
        "semantic-paraphrase",
        "How can a system refuse to answer when evidence is insufficient?",
    ),
    (
        "semantic-paraphrase",
        "How can differently worded questions retrieve the same passage?",
    ),
    (
        "multi-concept",
        "Why use a reranker after lexical and dense hybrid retrieval?",
    ),
    (
        "multi-concept",
        "How do chunk size and overlap affect retrieval and generation?",
    ),
    (
        "hard-negative",
        "Does a higher cosine score always mean the document is relevant?",
    ),
    (
        "hard-negative",
        "Does every model-related document explain hallucination prevention?",
    ),
    (
        "no-answer",
        "Which Qdrant collection is deployed by SearchLab in production?",
    ),
    (
        "no-answer",
        "What Recall@10 did SearchLab HNSW achieve on the final benchmark?",
    ),
]


def main() -> None:
    query_rows = _read_jsonl(SOURCE_PROFILE / "queries.jsonl")
    query_map = {str(row["id"]): str(row["text"]) for row in query_rows}
    qrels = _read_qrels(SOURCE_PROFILE / "qrels.tsv")
    id_map = json.loads(
        (CORPUS_ROOT / "id_maps" / "scifact.json").read_text(
            encoding="utf-8"
        )
    )
    qrels_by_query: dict[str, dict[str, int]] = {}
    for query_id, original_document_id, relevance in qrels:
        chunk_id = id_map.get(original_document_id)
        if chunk_id is None:
            raise ValueError(
                f"SciFact qrel document missing from canonical corpus: "
                f"{original_document_id}"
            )
        qrels_by_query.setdefault(query_id, {})[chunk_id] = relevance

    selected_ids = sorted(qrels_by_query)[:REVIEWED_QUERY_COUNT]
    reviewed = [
        {
            "query_id": f"scifact-q-{query_id}",
            "query": query_map[query_id],
            "category": classify_query(query_map[query_id]),
            "language": "en",
            "relevance": dict(sorted(qrels_by_query[query_id].items())),
            "notes": (
                "Relevance labels come from the human-annotated "
                "BEIR SciFact test qrels."
            ),
            "reviewed": True,
            "review_source": "BEIR SciFact human qrels",
            "source_query_id": query_id,
        }
        for query_id in selected_ids
    ]
    proposed = [
        {
            "query_id": f"draft-q-{index:03d}",
            "query": query,
            "category": category,
            "language": "en",
            "relevance": {},
            "notes": (
                "LLM-assisted candidate for a future SearchLab-domain "
                "quality corpus. It is excluded from formal metrics."
            ),
            "reviewed": False,
            "review_source": None,
            "source_query_id": None,
        }
        for index, (category, query) in enumerate(
            PROPOSED_UNREVIEWED,
            start=1,
        )
    ]
    draft = reviewed + proposed
    if len(draft) != 50:
        raise ValueError("retrieval query draft must contain exactly 50 rows")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    draft_path = OUTPUT_ROOT / "retrieval_queries_draft.jsonl"
    reviewed_path = OUTPUT_ROOT / "retrieval_queries.jsonl"
    _write_jsonl(draft_path, draft)
    _write_jsonl(reviewed_path, reviewed)
    manifest = {
        "version": "retrieval-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_version": "quality-v1",
        "draft_queries": len(draft),
        "reviewed_queries": len(reviewed),
        "unreviewed_queries": len(proposed),
        "formal_metrics_use_reviewed_only": True,
        "relevance_scale": {
            "3": "highly relevant",
            "2": "relevant evidence",
            "1": "weak/background relevance or binary benchmark relevance",
            "0": "not relevant",
        },
        "label_provenance": (
            "Formal labels are the public human-annotated BEIR SciFact "
            "test qrels. Proposed SearchLab-domain queries remain unreviewed."
        ),
        "files": {
            draft_path.name: _file_record(draft_path),
            reviewed_path.name: _file_record(reviewed_path),
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(draft)} draft queries and "
        f"{len(reviewed)} reviewed queries."
    )


def classify_query(query: str) -> str:
    if re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", query):
        return "acronym"
    lowered = query.lower()
    if " and " in lowered or " or " in lowered or "associated with" in lowered:
        return "multi-concept"
    if re.search(r"\d|[-/]", query):
        return "exact-term"
    return "semantic-paraphrase"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_qrels(path: Path) -> list[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            (
                row["query_id"],
                row["document_id"],
                int(row["relevance"]),
            )
            for row in reader
            if int(row["relevance"]) > 0
        ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    main()
