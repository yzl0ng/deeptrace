from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from app.core.context_builder import ContextBuilder
from app.core.llm import DeepSeekClient, DeepSeekSettings
from app.core.rag import RAGService
from app.evaluation.failure_analysis import FailureAnalyzer, FailureType
from app.evaluation.retrieval_metrics import ndcg_at_k, reciprocal_rank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "experiments" / "failure-analysis-v1"
)
RETRIEVAL_INPUT = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "retrieval-quality-v1"
    / "per_query.jsonl"
)
BASELINE_RAG_INPUT = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "rag-baseline-v1"
    / "responses.jsonl"
)
SELECTED_RETRIEVAL_IDS = (
    "scifact-q-1221",
    "scifact-q-1241",
    "scifact-q-1259",
    "scifact-q-1379",
    "scifact-q-1395",
    "scifact-q-312",
    "scifact-q-577",
    "scifact-q-644",
    "scifact-q-527",
    "scifact-q-183",
)
RAG_QUERIES = (
    {
        "query_id": "rag-q-001",
        "query": "如何避免模型产生幻觉",
        "category": "multi-concept",
        "answerable": True,
        "relevant_chunk_ids": ["doc-024", "doc-008"],
    },
    {
        "query_id": "rag-q-002",
        "query": "RRF 为什么不能直接相加 BM25 和 cosine 分数",
        "category": "exact-term",
        "answerable": True,
        "relevant_chunk_ids": ["doc-003"],
    },
    {
        "query_id": "rag-q-003",
        "query": "ANN Recall 和 Retrieval Recall 有什么区别",
        "category": "multi-concept",
        "answerable": True,
        "relevant_chunk_ids": ["doc-012", "doc-019", "doc-029"],
    },
    {
        "query_id": "rag-q-004",
        "query": "HNSW 的 efSearch 如何影响搜索",
        "category": "exact-term",
        "answerable": True,
        "relevant_chunk_ids": ["doc-013", "doc-005"],
    },
    {
        "query_id": "rag-q-005",
        "query": "SearchLab 的创始人是谁，他是哪一年出生的",
        "category": "no-answer",
        "answerable": False,
        "relevant_chunk_ids": [],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic failure candidates from saved SearchLab "
            "experiments and an optional bounded live RAG comparison."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-live-rag",
        action="store_true",
        help=(
            "Run five real baseline/reranked DeepSeek comparisons. Without "
            "this flag, reuse the saved rag_comparison.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "rag_comparison.jsonl"
    generated_at = datetime.now().astimezone().isoformat()

    if args.run_live_rag:
        live_records = run_live_rag_comparison()
        write_jsonl(comparison_path, live_records)
    elif comparison_path.is_file():
        live_records = read_jsonl(comparison_path)
    else:
        live_records = []

    historical_baseline = read_jsonl(BASELINE_RAG_INPUT)
    retrieval_rows = {
        row["query_id"]: row for row in read_jsonl(RETRIEVAL_INPUT)
    }
    raw_cases = [
        retrieval_case(retrieval_rows[query_id], index)
        for index, query_id in enumerate(SELECTED_RETRIEVAL_IDS, start=1)
    ]
    raw_cases.extend(
        rag_case(record, len(raw_cases) + index)
        for index, record in enumerate(live_records, start=1)
    )

    analyzer = FailureAnalyzer()
    candidates = analyzer.analyze_many(raw_cases)
    baseline_summary, reranked_summary = summarize_rag(live_records)
    summary = analyzer.summarize(
        candidates,
        experiment_id="failure-analysis-v1",
        generated_at=generated_at,
        corpus_version="scifact-quality-v1 + searchlab-demo-30",
        query_set_version="retrieval-v1-selected + rag-ablation-v1",
        baseline_rag=baseline_summary,
        reranked_rag=reranked_summary,
    )
    config = {
        "experiment_id": "failure-analysis-v1",
        "generated_at": generated_at,
        "inputs": {
            "retrieval_per_query": relative(RETRIEVAL_INPUT),
            "baseline_rag_v1": relative(BASELINE_RAG_INPUT),
            "live_rag_comparison": relative(comparison_path),
        },
        "selection": {
            "retrieval_query_ids": list(SELECTED_RETRIEVAL_IDS),
            "rag_query_ids": [item["query_id"] for item in RAG_QUERIES],
            "historical_baseline_records_read": len(historical_baseline),
            "selection_rule": (
                "Ten saved benchmark cases covering single-path and RRF "
                "wins/regressions, plus the five pre-existing RAG baseline "
                "queries rerun with the current shared prompt."
            ),
        },
        "pipelines": {
            "baseline": "RRF Top 5 -> DeepSeek",
            "reranked": (
                "RRF Top 20 -> BAAI/bge-reranker-v2-m3 Top 5 -> DeepSeek"
            ),
        },
        "parameters": {
            "retrieval_top_k": 5,
            "candidate_k": 20,
            "rank_constant": 60,
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": os.getenv(
                "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
            ),
            "generation_model": os.getenv(
                "DEEPSEEK_MODEL", "deepseek-v4-flash"
            ),
            "temperature": 0,
            "cost": None,
        },
        "review_policy": (
            "SciFact cases retain benchmark reviewed=true because human qrels "
            "support relevance comparisons. Demo RAG cases and all semantic "
            "generation/citation judgments remain reviewed=false."
        ),
    }

    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "failure_candidates.jsonl", candidates)
    write_jsonl(
        output_dir / "failure_cases.jsonl",
        [item for item in candidates if item["reviewed"]],
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(config, summary, candidates),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "queries_analyzed": summary["queries_analyzed"],
                "reviewed": summary["reviewed_cases"],
                "unreviewed": summary["unreviewed_candidates"],
                "reranker_improvements": summary[
                    "reranker_improvements"
                ],
                "reranker_regressions": summary[
                    "reranker_regressions"
                ],
                "output": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


def run_live_rag_comparison() -> list[dict[str, Any]]:
    # Imports are delayed so ordinary report generation and tests never load
    # BGE, the reranker, or application-global indices.
    from app.main import index_manager, reranker_runtime

    settings = DeepSeekSettings.from_environment()
    client = DeepSeekClient(settings)
    snapshot = index_manager.current()
    baseline_service = RAGService(
        snapshot.hybrid,
        client,
        ContextBuilder(),
    )
    reranked_service = RAGService(
        snapshot.hybrid,
        client,
        ContextBuilder(),
        reranker=reranker_runtime,
    )
    records: list[dict[str, Any]] = []
    for spec in RAG_QUERIES:
        query = str(spec["query"])
        bm25 = snapshot.bm25.search(query, top_k=20)
        dense = snapshot.dense.search(query, top_k=20)
        rrf = snapshot.hybrid.search(
            query,
            top_k=20,
            candidate_k=20,
            rank_constant=60,
        )
        baseline = baseline_service.answer(
            query,
            retrieval_top_k=5,
            candidate_k=20,
            rank_constant=60,
        )
        reranked = reranked_service.answer(
            query,
            retrieval_top_k=5,
            candidate_k=20,
            rank_constant=60,
        )
        records.append(
            {
                **spec,
                "retrieval": {
                    "bm25": {
                        "ranked_chunk_ids": [
                            item.document.id for item in bm25.hits
                        ],
                        "scores": [item.score for item in bm25.hits],
                        "latency_ms": bm25.elapsed_ms,
                    },
                    "dense": {
                        "ranked_chunk_ids": [
                            item.document.id for item in dense.hits
                        ],
                        "scores": [item.score for item in dense.hits],
                        "latency_ms": dense.elapsed_ms,
                    },
                    "rrf": {
                        "ranked_chunk_ids": [
                            item.document.id for item in rrf.hits
                        ],
                        "scores": [item.rrf_score for item in rrf.hits],
                        "latency_ms": rrf.elapsed_ms,
                    },
                    "reranker": {
                        "before": [
                            item.document.id
                            for item in sorted(
                                rrf.hits, key=lambda hit: hit.rank
                            )
                        ],
                        "after": [
                            trace.document_id
                            for trace in sorted(
                                (
                                    reranked.retrieval.reranking.traces
                                    if reranked.retrieval.reranking
                                    else []
                                ),
                                key=lambda trace: trace.rerank_rank,
                            )
                        ],
                        "rank_changes": {
                            trace.document_id: {
                                "rrf_rank": trace.rrf_rank,
                                "rerank_rank": trace.rerank_rank,
                                "rank_delta": trace.rank_delta,
                                "score": trace.reranker_score,
                            }
                            for trace in (
                                reranked.retrieval.reranking.traces
                                if reranked.retrieval.reranking
                                else []
                            )
                        },
                    },
                },
                "baseline": rag_variant(baseline),
                "reranked": rag_variant(reranked),
            }
        )
        print(
            f"{spec['query_id']}: baseline={baseline.abstained}, "
            f"reranked={reranked.abstained}, "
            f"reranker_ms={reranked.latency.reranker_ms:.2f}"
        )
    return records


def retrieval_case(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    retrieval = row["retrievers"]
    relevance = row["relevance"]
    return {
        "case_id": f"failure-{index:03d}",
        "query_id": row["query_id"],
        "query": row["query"],
        "query_category": row["category"],
        "corpus_version": "scifact-quality-v1",
        "expected": {
            "answerable": None,
            "required_facts": [],
            "relevance": relevance,
            "relevant_chunk_ids": list(relevance),
        },
        "retrieval": {
            name: {
                **details,
                "relevant_ranks": relevant_ranks(
                    details["ranked_chunk_ids"], relevance
                ),
            }
            for name, details in retrieval.items()
        },
        "rag": {},
        "metrics": {
            name: details["metrics"]
            for name, details in retrieval.items()
        },
        "evidence": [
            "SciFact human qrels identify relevant chunk IDs; rankings and "
            "metrics are read from retrieval-quality-v1/per_query.jsonl."
        ],
        "reviewed": True,
        "review_notes": (
            "Relevance is benchmark-reviewed. The automatic failure label and "
            "proposed causal explanation have not received a separate human audit."
        ),
    }


def rag_case(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    relevance = {
        item_id: 1 for item_id in record["relevant_chunk_ids"]
    }
    rag = {
        "baseline": record["baseline"],
        "reranked": record["reranked"],
    }
    return {
        "case_id": f"failure-{index:03d}",
        "query_id": record["query_id"],
        "query": record["query"],
        "query_category": record["category"],
        "corpus_version": "searchlab-demo-30",
        "expected": {
            "answerable": record["answerable"],
            "required_facts": [],
            "relevance": relevance,
            "relevant_chunk_ids": list(relevance),
        },
        "retrieval": record["retrieval"],
        "rag": rag,
        "metrics": {
            "baseline": retrieval_metrics(
                record["retrieval"]["reranker"]["before"][:5], relevance
            ),
            "reranked": retrieval_metrics(
                record["retrieval"]["reranker"]["after"][:5], relevance
            ),
            "rag_comparison": FailureAnalyzer().compare_rag_variants(rag),
        },
        "evidence": [
            "Ranks, answers, citations, latency and token usage come from the "
            "saved bounded live comparison; expected demo IDs are project "
            "fixtures and have not been independently human-reviewed."
        ],
        "reviewed": False,
        "review_notes": (
            "Automatic candidate only. Answer correctness, faithfulness and "
            "citation support require human review."
        ),
    }


def rag_variant(response: Any) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    return {
        "answer": payload["answer"],
        "model": payload["model"],
        "context_chunk_ids": [
            item["document"]["id"] for item in payload["retrieval"]["hits"]
        ],
        "context_hits": [
            {
                "rank": item["rank"],
                "document_id": item["document"]["id"],
                "title": item["document"]["title"],
                "content": item["document"]["content"],
                "rrf_rank": item.get("rrf_rank"),
                "reranker_score": item.get("reranker_score"),
            }
            for item in payload["retrieval"]["hits"]
        ],
        "citations": [
            item["citation_id"] for item in payload["citations"]
        ],
        "invalid_citation_ids": payload["invalid_citation_ids"],
        "abstained": payload["abstained"],
        "abstention_reason": payload["abstention_reason"],
        "context": payload["context"],
        "latency": payload["latency"],
        "usage": payload["usage"],
        "prompt": payload["prompt"],
    }


def summarize_rag(
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def variant(name: str) -> dict[str, Any]:
        if not records:
            return {}
        judged = [
            row for row in records if row.get("relevant_chunk_ids")
        ]
        ranking_key = "before" if name == "baseline" else "after"
        return {
            "queries": len(records),
            "judged_retrieval_queries": len(judged),
            "mean_mrr": mean(
                reciprocal_rank(
                    row["retrieval"]["reranker"][ranking_key][:5],
                    {
                        item_id: 1
                        for item_id in row["relevant_chunk_ids"]
                    },
                )
                for row in judged
            ),
            "mean_ndcg@5": mean(
                ndcg_at_k(
                    row["retrieval"]["reranker"][ranking_key][:5],
                    {
                        item_id: 1
                        for item_id in row["relevant_chunk_ids"]
                    },
                    5,
                )
                for row in judged
            ),
            "mean_total_latency_ms": mean(
                row[name]["latency"]["total_ms"] for row in records
            ),
            "mean_generation_latency_ms": mean(
                row[name]["latency"]["generation_ms"] for row in records
            ),
            "mean_reranker_latency_ms": mean(
                (row[name]["latency"].get("reranker_ms") or 0)
                for row in records
            ),
            "total_tokens": sum(
                row[name]["usage"]["total_tokens"] for row in records
            ),
            "abstentions": sum(
                bool(row[name]["abstained"]) for row in records
            ),
            "invalid_citations": sum(
                len(row[name]["invalid_citation_ids"]) for row in records
            ),
            "cost": None,
        }

    return variant("baseline"), variant("reranked")


def render_report(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> str:
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        for label in candidate["failure_types"]:
            by_label.setdefault(label, []).append(candidate)

    def case_block(label: str, title: str) -> list[str]:
        cases = by_label.get(label, [])
        if not cases:
            return [f"## {title}", "", "`not_observed`", ""]
        case = cases[0]
        retrieval = case.get("retrieval", {})
        rag = case.get("rag", {})
        lines = [
            f"## {title}",
            "",
            f"- Query：`{case['query_id']}` {case['query']}",
            f"- Reviewed：`{str(case['reviewed']).lower()}`",
            f"- 首个候选层：`{case['pipeline_stages'][0] if case['pipeline_stages'] else 'unknown'}`",
            f"- 标签：{', '.join(f'`{item}`' for item in case['failure_types'])}",
            f"- 证据：{' '.join(case['evidence'][:3])}",
            f"- 修复：{' '.join(case['possible_fix'])}",
            f"- 代价：{case['tradeoff']}",
        ]
        for method in ("bm25", "dense", "dense_exact", "rrf"):
            if method in retrieval:
                lines.append(
                    f"- {method} Top-5："
                    f"`{retrieval[method].get('ranked_chunk_ids', [])[:5]}`"
                )
        if rag:
            lines.extend(
                [
                    f"- Baseline answer：{short(rag['baseline']['answer'])}",
                    f"- Reranked answer：{short(rag['reranked']['answer'])}",
                    f"- Baseline citations：`{rag['baseline']['citations']}`",
                    f"- Reranked citations：`{rag['reranked']['citations']}`",
                ]
            )
        lines.append("")
        return lines

    lines = [
        "# SearchLab failure-analysis-v1",
        "",
        "> 自动标签只用于定位人工复核候选；除 SciFact qrels 支持的检索比较外，",
        "> 不把生成正确性、忠实度、引用正确性或语料缺失描述为已证明结论。",
        "",
        "## 实验配置与数据来源",
        "",
        f"- Generated：`{config['generated_at']}`",
        f"- 查询：{summary['queries_analyzed']}（reviewed "
        f"{summary['reviewed_cases']} / unreviewed "
        f"{summary['unreviewed_candidates']}）",
        f"- Retrieval：`{config['inputs']['retrieval_per_query']}`",
        f"- 历史基础 RAG：`{config['inputs']['baseline_rag_v1']}`",
        f"- 当前公平对照：`{config['inputs']['live_rag_comparison']}`",
        f"- Query set：`{summary['query_set_version']}`",
        "",
        "## 两条 RAG 链路",
        "",
        f"- Without Reranker：{config['pipelines']['baseline']}",
        f"- With Reranker：{config['pipelines']['reranked']}",
        "- 两路使用当前相同 Grounded Prompt、DeepSeek 配置与 demo-30 语料。",
        "",
        "## 错误分析决策树",
        "",
        "Corpus → Parsing → Chunking → BM25 → Dense → RRF → Reranker → "
        "Context → Generation → Citation → Abstention → Security。",
        "定位最早出现的可观察信号；下游无法补回 RRF candidate 之外的证据。",
        "",
        "## 错误类型与 Pipeline 统计",
        "",
        "| Failure type | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{label}` | {count} |"
        for label, count in summary["failure_counts"].items()
    )
    lines.extend(["", "| Stage | Count |", "| --- | ---: |"])
    lines.extend(
        f"| `{stage}` | {count} |"
        for stage, count in summary["pipeline_stage_counts"].items()
    )
    lines.extend([""])
    lines.extend(case_block("dense_false_positive", "BM25 成功 / Dense 失败"))
    lines.extend(case_block("lexical_mismatch", "Dense 成功 / BM25 失败"))
    lines.extend(case_block("fusion_improvement", "RRF 改善案例"))
    lines.extend(case_block("fusion_regression", "RRF 回退案例"))
    lines.extend(
        case_block("reranker_improvement", "Reranker 改善 Top-5")
    )
    lines.extend(
        case_block("reranker_regression", "Reranker 回退 Top-5")
    )
    lines.extend(
        case_block(
            "candidate_missing_before_rerank",
            "候选缺失，Reranker 无法修复",
        )
    )
    lines.extend(
        case_block(
            "generation_hallucination",
            "检索正确但生成失败",
        )
    )
    lines.extend(case_block("citation_invalid_id", "引用非法 ID"))
    lines.extend(case_block("citation_missing", "引用缺失"))
    lines.extend(case_block("correct_abstention", "正确拒答"))
    lines.extend(
        case_block("no_answer_false_positive", "应该拒答但未拒答")
    )
    lines.extend(case_block("false_abstention", "错误拒答"))
    lines.extend(case_block("parsing_failure", "上传、解析或 Chunking"))
    lines.extend(
        case_block("prompt_injection_risk", "Prompt Injection 风险")
    )
    lines.extend(
        [
            "## Without / With Reranker 汇总",
            "",
            "| Variant | MRR* | nDCG@5* | Mean total ms | Mean generation ms | "
            "Mean reranker ms | Tokens | Abstentions | Invalid citations | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            rag_summary_row("Without", summary["baseline_rag"]),
            rag_summary_row("With", summary["reranked_rag"]),
            "",
            "这些数字描述延迟、token、引用 ID 合法性和拒答行为；没有人工答案标注，",
            "因此不计算 Answer Correctness、Faithfulness 或“质量提升百分比”。",
            "",
            "## 未观察到的类型",
            "",
            (
                ", ".join(
                    f"`{item}`" for item in summary["no_observed_case"]
                )
                or "无"
            ),
            "",
            "## 当前限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(
        [
            "- SciFact 检索案例与 SearchLab demo RAG 案例属于不同数据分布，"
            "不合并成单一质量分数。",
            "- 当前没有 OCR、分布式任务队列、HNSW/Qdrant 或生产级权限系统。",
            "",
            "## 可以用于面试的结论",
            "",
            "- 能用真实 qrels 展示 BM25 与 Dense 的互补，以及 RRF 的改善和回退。",
            "- 能展示真实 Reranker 排名 trace、延迟代价和候选集边界。",
            "- 能展示无答案问题的拒答、服务端引用 ID 验证与未观察到案例。",
            "",
            "## 仍不能写进简历的结论",
            "",
            "- 不能声称 Reranker、RAG、Faithfulness 或 Citation Correctness "
            "整体显著提升。",
            "- 不能把自动候选、LLM 判断或存在引用当作人工 ground truth。",
            "- 不能声称已实现 HNSW/Qdrant、OCR、分布式队列或完整 Prompt "
            "Injection 防御。",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def retrieval_metrics(
    ranked_ids: list[str], relevance: Mapping[str, int]
) -> dict[str, float]:
    return {
        "mrr": reciprocal_rank(ranked_ids, relevance),
        "ndcg@5": ndcg_at_k(ranked_ids, relevance, 5),
    }


def relevant_ranks(
    ranked_ids: Iterable[str], relevance: Mapping[str, int]
) -> dict[str, int]:
    relevant = {key for key, grade in relevance.items() if grade > 0}
    return {
        item_id: rank
        for rank, item_id in enumerate(ranked_ids, start=1)
        if item_id in relevant
    }


def rag_summary_row(name: str, values: Mapping[str, Any]) -> str:
    if not values:
        return (
            f"| {name} | not measured | — | — | — | — | — | — | — | null |"
        )
    return (
        f"| {name} | {values['mean_mrr']:.3f} | "
        f"{values['mean_ndcg@5']:.3f} | "
        f"{values['mean_total_latency_ms']:.2f} | "
        f"{values['mean_generation_latency_ms']:.2f} | "
        f"{values['mean_reranker_latency_ms']:.2f} | "
        f"{values['total_tokens']} | {values['abstentions']} | "
        f"{values['invalid_citations']} | null |"
    )


def short(value: str, limit: int = 240) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
