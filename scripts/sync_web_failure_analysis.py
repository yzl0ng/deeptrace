from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "data" / "experiments" / "failure-analysis-v1"
)
TARGET = PROJECT_ROOT / "web" / "app" / "failure-analysis-report.json"


def main() -> None:
    config = read_json(SOURCE / "config.json")
    summary = read_json(SOURCE / "summary.json")
    candidates = read_jsonl(SOURCE / "failure_candidates.jsonl")
    snapshot = {
        "config": {
            "experiment_id": config["experiment_id"],
            "generated_at": summary["generated_at"],
            "corpus_version": summary["corpus_version"],
            "query_set_version": summary["query_set_version"],
            "pipelines": config["pipelines"],
            "review_policy": config["review_policy"],
        },
        "summary": summary,
        "cases": [web_case(item) for item in candidates],
    }
    TARGET.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {TARGET.relative_to(PROJECT_ROOT)} "
        f"with {len(candidates)} saved cases."
    )


def web_case(item: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = item.get("retrieval", {})
    rag = item.get("rag", {})
    return {
        "case_id": item["case_id"],
        "query_id": item["query_id"],
        "query": item["query"],
        "query_category": item.get("query_category"),
        "corpus_version": item["corpus_version"],
        "reviewed": item["reviewed"],
        "review_notes": item["review_notes"],
        "failure_types": item["failure_types"],
        "pipeline_stages": item["pipeline_stages"],
        "expected": item["expected"],
        "retrieval": {
            name: compact_ranking(details)
            for name, details in retrieval.items()
        },
        "rag": {
            name: compact_rag(details)
            for name, details in rag.items()
        },
        "root_cause": item.get("root_cause"),
        "evidence": item["evidence"],
        "possible_fix": item.get(
            "possible_fix",
            ["本次没有触发确定性失败规则，保留为无变化对照。"],
        ),
        "tradeoff": item.get(
            "tradeoff",
            "没有人工质量标注，不能据此声称回答质量不变。",
        ),
    }


def compact_ranking(details: Mapping[str, Any]) -> dict[str, Any]:
    if "ranked_chunk_ids" in details:
        return {
            "ranked_chunk_ids": details["ranked_chunk_ids"][:5],
            "relevant_ranks": details.get("relevant_ranks", {}),
            "latency_ms": details.get("latency_ms"),
        }
    if "before" in details:
        changes = details.get("rank_changes", {})
        visible = set(details["before"][:5]) | set(details["after"][:5])
        return {
            "before": details["before"][:5],
            "after": details["after"][:5],
            "rank_changes": {
                key: value for key, value in changes.items() if key in visible
            },
        }
    return dict(details)


def compact_rag(details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answer": details.get("answer"),
        "context_chunk_ids": details.get("context_chunk_ids", []),
        "citations": details.get("citations", []),
        "invalid_citation_ids": details.get("invalid_citation_ids", []),
        "abstained": details.get("abstained"),
        "abstention_reason": details.get("abstention_reason"),
        "latency": details.get("latency", {}),
        "usage": details.get("usage", {}),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
