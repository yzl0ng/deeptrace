from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from app.agentic.trajectory import TrajectorySeed

SOURCES = (
    {
        "name": "hotpotqa",
        "repository": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "license": "CC-BY-SA-4.0",
    },
    {
        "name": "2wikimultihopqa",
        "repository": "framolfese/2WikiMultihopQA",
        "config": "default",
        "split": "train",
        "revision": "fe713bfbd1afbca1a65246741a75890405d56a3a",
        "license": "Apache-2.0",
    },
    {
        "name": "musique",
        "repository": "bdsaglam/musique",
        "config": "answerable",
        "split": "train",
        "revision": "22873a405dd809893b22ada0b499299fb612d2df",
        "license": "CC-BY-4.0",
    },
)


def _evidence_for_row(source: str, row: dict[str, Any]) -> list[dict[str, str]]:
    if source in {"hotpotqa", "2wikimultihopqa"}:
        context = row.get("context", {})
        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences = (
            context.get("sentences", []) if isinstance(context, dict) else []
        )
        supporting = row.get("supporting_facts", {})
        support_titles = set(
            supporting.get("title", [])
            if isinstance(supporting, dict)
            else []
        )
        evidence = []
        for index, (title, parts) in enumerate(zip(titles, sentences)):
            if support_titles and title not in support_titles:
                continue
            text = " ".join(str(part) for part in parts).strip()
            if text:
                evidence.append(
                    {
                        "evidence_id": f"{source}-e{index}",
                        "title": str(title),
                        "text": text[:2000],
                    }
                )
        return evidence[:8]

    evidence = []
    for index, paragraph in enumerate(row.get("paragraphs", [])):
        if not paragraph.get("is_supporting", False):
            continue
        text = str(paragraph.get("paragraph_text", "")).strip()
        if text:
            evidence.append(
                {
                    "evidence_id": f"{source}-e{index}",
                    "title": str(paragraph.get("title", "")),
                    "text": text[:2000],
                }
            )
    return evidence[:8]


def seed_from_row(
    source: str, row: dict[str, Any], index: int
) -> TrajectorySeed:
    upstream_id = str(row.get("id") or row.get("_id") or index)
    return TrajectorySeed(
        seed_id=f"{source}-train-{upstream_id}",
        query=str(row["question"]).strip(),
        language="en",
        variant="research" if index % 2 == 0 else "verification",
        expected_answer=str(row.get("answer", "")).strip(),
        evidence=_evidence_for_row(source, row),
    )


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a pinned 500-record trajectory seed pool."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-source", type=int, default=167)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    if args.records_per_source < 1 or args.limit < 1:
        parser.error("record counts must be positive")

    from datasets import load_dataset

    seeds: list[TrajectorySeed] = []
    source_counts: dict[str, int] = {}
    for source in SOURCES:
        dataset = load_dataset(
            source["repository"],
            source["config"],
            split=source["split"],
            revision=source["revision"],
            streaming=True,
        )
        rows = list(islice(dataset, args.records_per_source))
        source_counts[source["name"]] = len(rows)
        seeds.extend(
            seed_from_row(source["name"], row, index)
            for index, row in enumerate(rows)
        )

    seeds = seeds[: args.limit]
    if len(seeds) != args.limit:
        raise RuntimeError(f"requested {args.limit} seeds but built {len(seeds)}")
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise RuntimeError("seed IDs are not unique")
    if len({seed.query for seed in seeds}) != len(seeds):
        raise RuntimeError("seed queries are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "seeds.jsonl"
    content = _jsonl(seed.model_dump(mode="json") for seed in seeds)
    seed_path.write_bytes(content)
    manifest = {
        "dataset_version": "trajectory-seeds-500-v1",
        "records": len(seeds),
        "unique_ids": len({seed.seed_id for seed in seeds}),
        "unique_queries": len({seed.query for seed in seeds}),
        "selection": (
            "First records from each pinned training split, capped to 500 "
            "after source-order concatenation."
        ),
        "source_counts_before_cap": source_counts,
        "sources": list(SOURCES),
        "artifact": {
            "path": seed_path.name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        "truth_boundary": (
            "Seeds come from public benchmark training splits and are not part "
            "of the frozen six-case test subset. Teacher trajectories remain "
            "automatically generated and require quality review."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
