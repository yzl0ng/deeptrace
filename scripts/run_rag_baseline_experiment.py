from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

from app.core.context_builder import ContextBuilder
from app.core.llm import DeepSeekClient, DeepSeekSettings
from app.core.rag import RAGService
from app.main import hybrid_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "experiments" / "rag-baseline-v1"
)
QUERIES = [
    "如何避免模型产生幻觉",
    "RRF 为什么不能直接相加 BM25 和 cosine 分数",
    "ANN Recall 和 Retrieval Recall 有什么区别",
    "HNSW 的 efSearch 如何影响搜索",
    "SearchLab 的创始人是谁，他是哪一年出生的",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain DeepSeek with the no-reranker grounded RAG baseline."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = DeepSeekSettings.from_environment()
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
    client = DeepSeekClient(settings)
    rag = RAGService(
        hybrid_retriever,
        client,
        ContextBuilder(),
        max_context_chars=max_context_chars,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat()
    config = {
        "experiment_id": "rag-baseline-v1",
        "corpus": "searchlab-demo-30",
        "retriever": "rrf",
        "retrieval_top_k": 5,
        "candidate_k": 20,
        "rank_constant": 60,
        "embedding_model": "BAAI/bge-m3",
        "generation_model": settings.model,
        "temperature": 0,
        "reranker": None,
        "max_context_chars": max_context_chars,
        "comparison": "plain_deepseek_vs_grounded_rag",
        "generated_at": generated_at,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, object]] = []
    for index, query in enumerate(QUERIES, start=1):
        plain_started = time.perf_counter()
        plain = client.generate_plain(query=query)
        plain_ms = (time.perf_counter() - plain_started) * 1000
        grounded = rag.answer(
            query,
            retrieval_top_k=5,
            candidate_k=20,
            rank_constant=60,
        )
        record = {
            "query_id": f"rag-q-{index:03d}",
            "query": query,
            "plain_deepseek": {
                "answer": plain.text,
                "model": plain.model,
                "latency_ms": plain_ms,
                "usage": {
                    "prompt_tokens": plain.usage.prompt_tokens,
                    "completion_tokens": plain.usage.completion_tokens,
                    "total_tokens": plain.usage.total_tokens,
                },
            },
            "grounded_rag": grounded.model_dump(mode="json"),
        }
        records.append(record)
        print(f"\n[{index}/{len(QUERIES)}] {query}")
        print("\n--- Plain DeepSeek ---")
        print(plain.text)
        print("\n--- Grounded RAG (RRF Top 5, no Reranker) ---")
        print(grounded.answer)
        print(
            "citations="
            f"{[item.citation_id for item in grounded.citations]} "
            f"invalid={grounded.invalid_citation_ids} "
            f"abstained={grounded.abstained}"
        )

    with (output_dir / "responses.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(
        render_report(config, records),
        encoding="utf-8",
    )


def render_report(
    config: dict[str, object],
    records: list[dict[str, object]],
) -> str:
    lines = [
        "# rag-baseline-v1",
        "",
        "本实验并排记录普通 DeepSeek 与无 Reranker 的 Grounded RAG。",
        "这些案例用于定性错误分析，不构成 Answer Correctness、Faithfulness",
        "或 Citation Correctness 的正式提升数字。",
        "",
        f"- 生成时间：{config['generated_at']}",
        f"- 模型：{config['generation_model']}",
        "- RAG：RRF Top 5 → DeepSeek",
        "- Reranker：无",
        "",
    ]
    for record in records:
        plain = record["plain_deepseek"]
        grounded = record["grounded_rag"]
        citation_ids = [
            item["citation_id"] for item in grounded["citations"]
        ]
        lines.extend(
            [
                f"## {record['query_id']} · {record['query']}",
                "",
                "### Plain DeepSeek",
                "",
                str(plain["answer"]),
                "",
                "### Grounded RAG",
                "",
                str(grounded["answer"]),
                "",
                f"- 引用：{citation_ids}",
                f"- 无效引用：{grounded['invalid_citation_ids']}",
                f"- 拒答：{grounded['abstained']} / "
                f"{grounded['abstention_reason']}",
                f"- Plain latency：{plain['latency_ms']:.2f} ms",
                f"- RAG total latency："
                f"{grounded['latency']['total_ms']:.2f} ms",
                f"- Plain tokens：{plain['usage']['total_tokens']}",
                f"- RAG tokens：{grounded['usage']['total_tokens']}",
                "",
            ]
        )
    return "\n".join(line.rstrip() for line in lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
