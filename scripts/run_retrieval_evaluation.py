from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.bm25 import BM25Index
from app.core.dense import DenseSettings
from app.core.hybrid import HybridRetriever
from app.evaluation.corpus import CorpusChunk
from app.evaluation.evaluator import (
    EvaluationQuery,
    evaluate_retriever,
)
from app.evaluation.exact_dense import CachedExactDenseIndex
from app.models import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_MANIFEST_PATH = PROJECT_ROOT / "data" / "corpora" / "manifest.json"
QUERY_MANIFEST_PATH = PROJECT_ROOT / "data" / "evaluation" / "manifest.json"
QUERY_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_queries.jsonl"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "retrieval-quality-v1"
)
REPORT_SNAPSHOT_PATH = (
    PROJECT_ROOT / "data" / "reports" / "retrieval_quality_baseline.json"
)
CACHE_ROOT = PROJECT_ROOT / "data" / "index" / "evaluation"
TOP_K = 10
CANDIDATE_K = 20
RANK_CONSTANT = 60
FAILURE_LABELS = (
    "corpus_missing",
    "lexical_mismatch",
    "dense_false_positive",
    "acronym_failure",
    "chunk_too_short",
    "chunk_too_long",
    "duplicate_content",
    "fusion_regression",
    "annotation_uncertain",
    "no_answer_false_positive",
)


def main() -> None:
    corpus_manifest = _read_json(CORPUS_MANIFEST_PATH)
    query_manifest = _read_json(QUERY_MANIFEST_PATH)
    quality_path = (
        PROJECT_ROOT / corpus_manifest["quality_corpus"]["path"]
    )
    chunks = _load_chunks(quality_path)
    documents = [
        Document(
            id=chunk.chunk_id,
            title=chunk.title,
            content=chunk.text,
            source=chunk.source,
            metadata={
                **chunk.metadata,
                "topic": chunk.topic,
                "language": chunk.language,
                "content_hash": chunk.content_hash,
            },
        )
        for chunk in chunks
    ]
    query_rows = _read_jsonl(QUERY_PATH)
    if any(not row.get("reviewed") for row in query_rows):
        raise ValueError("formal evaluation cannot include unreviewed queries")
    queries = [
        EvaluationQuery(
            query_id=row["query_id"],
            query=row["query"],
            category=row["category"],
            relevance=row["relevance"],
            reviewed=bool(row["reviewed"]),
        )
        for row in query_rows
    ]

    lexical = BM25Index()
    lexical.build(documents)
    dense_settings = DenseSettings.from_environment()
    dense = CachedExactDenseIndex(
        documents,
        corpus_hash=corpus_manifest["quality_corpus"]["content_hash"],
        settings=dense_settings,
        cache_root=CACHE_ROOT,
    )
    hybrid = HybridRetriever(
        lexical,
        dense,
        rank_constant=RANK_CONSTANT,
        candidate_k=CANDIDATE_K,
    )

    summaries: dict[str, Any] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    for name, retriever in (
        ("bm25", lexical),
        ("dense_exact", dense),
        ("rrf", hybrid),
    ):
        if name in {"dense_exact", "rrf"}:
            dense.clear_query_cache()
        summary, per_query = evaluate_retriever(
            name,
            retriever,
            queries,
            top_k=TOP_K,
        )
        summaries[name] = summary
        results[name] = per_query

    generated_at = datetime.now(UTC).isoformat()
    cache_metadata = dense.cache_metadata()
    cache_metadata["cache_dir"] = str(
        Path(str(cache_metadata["cache_dir"])).relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    config = {
        "experiment_id": "retrieval-quality-v1",
        "generated_at": generated_at,
        "corpus_version": corpus_manifest["quality_corpus"]["version"],
        "corpus_hash": corpus_manifest["quality_corpus"]["content_hash"],
        "quality_corpus_documents": len(documents),
        "quality_corpus_chunks": len(chunks),
        "query_set_version": query_manifest["version"],
        "reviewed_queries": len(queries),
        "retrievers": ["bm25", "dense_exact", "rrf"],
        "embedding_model": dense.embedder.model_name,
        "device": dense.embedder.device,
        "vector_dimension": dense.vector_dimension,
        "top_k": TOP_K,
        "parameters": {
            "bm25": lexical.stats(),
            "dense_exact": {
                "normalized": True,
                "similarity": "exact NumPy dot product / cosine",
            },
            "rrf": {
                "rank_constant": RANK_CONSTANT,
                "candidate_k": CANDIDATE_K,
            },
        },
        "embedding_cache": cache_metadata,
        "recall_definitions": {
            "retrieval_recall": (
                "Human-qrel relevant chunks retrieved divided by all "
                "human-qrel relevant chunks."
            ),
            "ann_recall": (
                "HNSW Top-K overlap with exact dense Top-K; not measured "
                "in this experiment."
            ),
        },
    }
    merged = merge_per_query(queries, results)
    failures = classify_failures(merged, chunks)
    exact_ground_truth = [
        {
            "query_id": row["query_id"],
            "top_k": TOP_K,
            "results": [
                {
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": score,
                }
                for rank, (chunk_id, score) in enumerate(
                    zip(
                        row["ranked_chunk_ids"],
                        row["scores"],
                        strict=True,
                    ),
                    start=1,
                )
            ],
        }
        for row in results["dense_exact"]
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "config.json", config)
    _write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "experiment_id": config["experiment_id"],
            "generated_at": generated_at,
            "corpus_version": config["corpus_version"],
            "query_set_version": config["query_set_version"],
            "embedding_model": config["embedding_model"],
            "device": config["device"],
            "vector_dimension": config["vector_dimension"],
            "queries": len(queries),
            "summary": summaries,
        },
    )
    _write_jsonl(OUTPUT_ROOT / "per_query.jsonl", merged)
    _write_jsonl(OUTPUT_ROOT / "failures.jsonl", failures)
    _write_jsonl(
        OUTPUT_ROOT / "exact_ground_truth.jsonl",
        exact_ground_truth,
    )
    (OUTPUT_ROOT / "report.md").write_text(
        render_report(config, summaries, merged, failures),
        encoding="utf-8",
    )
    REPORT_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        REPORT_SNAPSHOT_PATH,
        {
            "config": config,
            "summary": summaries,
            "comparison_cases": comparison_cases(merged),
            "extreme_cases": extreme_cases(merged),
            "failure_counts": failure_counts(failures),
            "limitations": [
                (
                    "SciFact qrels are binary. The evaluator supports graded "
                    "relevance, but this baseline only exercises grade 1."
                ),
                (
                    "The corpus is English scientific text rather than a "
                    "Chinese SearchLab-domain benchmark."
                ),
                "HNSW, Qdrant, Reranker and LLM generation are not measured.",
            ],
        },
    )
    print(
        f"Evaluation complete: {len(queries)} reviewed queries, "
        f"device={dense.embedder.device}, "
        f"cache_hit={dense.cache_hit}"
    )


def merge_per_query(
    queries: list[EvaluationQuery],
    results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_method = {
        method: {row["query_id"]: row for row in rows}
        for method, rows in results.items()
    }
    return [
        {
            "query_id": query.query_id,
            "query": query.query,
            "category": query.category,
            "reviewed": query.reviewed,
            "relevance": dict(query.relevance),
            "retrievers": {
                method: {
                    "latency_ms": by_method[method][query.query_id][
                        "latency_ms"
                    ],
                    "ranked_chunk_ids": by_method[method][query.query_id][
                        "ranked_chunk_ids"
                    ],
                    "scores": by_method[method][query.query_id]["scores"],
                    "metrics": by_method[method][query.query_id]["metrics"],
                }
                for method in ("bm25", "dense_exact", "rrf")
            },
        }
        for query in queries
    ]


def classify_failures(
    rows: list[dict[str, Any]],
    chunks: list[CorpusChunk],
) -> list[dict[str, Any]]:
    chunk_lengths = {chunk.chunk_id: len(chunk.text) for chunk in chunks}
    failures: list[dict[str, Any]] = []
    for row in rows:
        bm25 = row["retrievers"]["bm25"]["metrics"]
        dense = row["retrievers"]["dense_exact"]["metrics"]
        rrf = row["retrievers"]["rrf"]["metrics"]
        labels: list[str] = []
        if not row["relevance"]:
            returned_anything = any(
                result["ranked_chunk_ids"]
                for result in row["retrievers"].values()
            )
            labels.append(
                "no_answer_false_positive"
                if returned_anything
                else "corpus_missing"
            )
        if bm25["recall@10"] < dense["recall@10"]:
            labels.append("lexical_mismatch")
        if dense["recall@10"] < bm25["recall@10"]:
            labels.append(
                "acronym_failure"
                if row["category"] == "acronym"
                else "dense_false_positive"
            )
        if rrf["ndcg@10"] < max(bm25["ndcg@10"], dense["ndcg@10"]):
            labels.append("fusion_regression")
        relevant_lengths = [
            chunk_lengths[chunk_id]
            for chunk_id in row["relevance"]
            if chunk_id in chunk_lengths
        ]
        if any(length < 200 for length in relevant_lengths):
            labels.append("chunk_too_short")
        if any(length > 4000 for length in relevant_lengths):
            labels.append("chunk_too_long")
        if not row["reviewed"]:
            labels.append("annotation_uncertain")
        if labels:
            failures.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "category": row["category"],
                    "failure_types": sorted(set(labels)),
                    "metrics": {
                        name: result["metrics"]
                        for name, result in row["retrievers"].items()
                    },
                }
            )
    return failures


def render_report(
    config: dict[str, Any],
    summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# SearchLab Retrieval Quality Baseline",
        "",
        f"- Generated: `{config['generated_at']}`",
        f"- Corpus: `{config['corpus_version']}` "
        f"({config['quality_corpus_chunks']} chunks)",
        f"- Query set: `{config['query_set_version']}` "
        f"({config['reviewed_queries']} reviewed queries)",
        f"- Embedding: `{config['embedding_model']}` / "
        f"`{config['device']}` / `{config['vector_dimension']} dim`",
        f"- Embedding cache hit: `{config['embedding_cache']['cache_hit']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Retriever | Recall@5 | Recall@10 | Precision@5 | MRR | "
        "nDCG@10 | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("bm25", "dense_exact", "rrf"):
        item = summaries[name]
        lines.append(
            f"| {name} | {item['recall@5']:.4f} | "
            f"{item['recall@10']:.4f} | "
            f"{item['precision@5']:.4f} | {item['mrr']:.4f} | "
            f"{item['ndcg@10']:.4f} | "
            f"{item['latency']['p50_ms']:.3f} | "
            f"{item['latency']['p95_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Retrieval Recall uses human benchmark qrels. ANN Recall is not "
            "reported here; it will later measure HNSW overlap with the saved "
            "exact dense Top-K.",
            "",
            "## Comparison cases",
            "",
        ]
    )
    comparisons = comparison_cases(rows)
    for label, cases in comparisons.items():
        lines.append(f"### {label}")
        lines.append("")
        if not cases:
            lines.append("No observed case in this query set.")
        for case in cases[:3]:
            lines.append(
                f"- `{case['query_id']}` {case['query']} "
                f"(BM25 nDCG@10={case['bm25']:.3f}, "
                f"Dense={case['dense']:.3f}, RRF={case['rrf']:.3f})"
            )
        lines.append("")
    lines.extend(["## Best and worst observed queries", ""])
    for method, cases in extreme_cases(rows).items():
        best = cases["best"]
        worst = cases["worst"]
        lines.append(
            f"- `{method}` best: `{best['query_id']}` "
            f"(nDCG@10={best['metrics']['ndcg@10']:.3f}); "
            f"worst: `{worst['query_id']}` "
            f"(nDCG@10={worst['metrics']['ndcg@10']:.3f})"
        )
    lines.extend(
        [
            "",
            "## Failure classification",
            "",
            f"- Queries with at least one observed failure label: "
            f"{len(failures)}",
        ]
    )
    for label, count in sorted(failure_counts(failures).items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- SciFact test qrels are binary, so the implementation supports "
            "graded relevance but this run does not exercise grades 2–3.",
            "- The corpus is English scientific text, not the final SearchLab "
            "RAG/search-domain corpus.",
            "- No HNSW, Qdrant, Reranker or LLM result is included.",
            "",
        ]
    )
    return "\n".join(lines)


def failure_counts(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in FAILURE_LABELS}
    for failure in failures:
        for label in failure["failure_types"]:
            counts[label] = counts.get(label, 0) + 1
    return counts


def extreme_cases(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("bm25", "dense_exact", "rrf"):
        ranked = sorted(
            rows,
            key=lambda row: (
                row["retrievers"][method]["metrics"]["ndcg@10"],
                row["query_id"],
            ),
        )
        result[method] = {
            "worst": {
                "query_id": ranked[0]["query_id"],
                "query": ranked[0]["query"],
                "metrics": ranked[0]["retrievers"][method]["metrics"],
            },
            "best": {
                "query_id": ranked[-1]["query_id"],
                "query": ranked[-1]["query"],
                "metrics": ranked[-1]["retrievers"][method]["metrics"],
            },
        }
    return result


def comparison_cases(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    cases = {
        "BM25 beats Dense": [],
        "Dense beats BM25": [],
        "RRF improves over both": [],
        "RRF regression": [],
    }
    for row in rows:
        bm25 = row["retrievers"]["bm25"]["metrics"]["ndcg@10"]
        dense = row["retrievers"]["dense_exact"]["metrics"]["ndcg@10"]
        rrf = row["retrievers"]["rrf"]["metrics"]["ndcg@10"]
        item = {
            "query_id": row["query_id"],
            "query": row["query"],
            "bm25": bm25,
            "dense": dense,
            "rrf": rrf,
        }
        if bm25 > dense:
            cases["BM25 beats Dense"].append(item)
        if dense > bm25:
            cases["Dense beats BM25"].append(item)
        if rrf > max(bm25, dense):
            cases["RRF improves over both"].append(item)
        if rrf < max(bm25, dense):
            cases["RRF regression"].append(item)
    for values in cases.values():
        values.sort(
            key=lambda item: abs(
                item["bm25"] - item["dense"]
            ),
            reverse=True,
        )
    return cases


def _load_chunks(path: Path) -> list[CorpusChunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            CorpusChunk.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
