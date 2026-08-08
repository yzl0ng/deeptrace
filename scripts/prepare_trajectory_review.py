from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.agentic.trajectory import (
    TrajectorySeed,
    parse_and_filter_trajectory,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source(seed_id: str) -> str:
    return seed_id.split("-train-", 1)[0]


def _sample_key(seed_id: str, sample_seed: str) -> str:
    return hashlib.sha256(f"{sample_seed}:{seed_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic private semantic-review packet."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-per-source", type=int, default=5)
    parser.add_argument("--rejected-limit", type=int, default=10)
    parser.add_argument("--sample-seed", default="20260729")
    args = parser.parse_args()
    if args.accepted_per_source < 1 or args.rejected_limit < 0:
        parser.error(
            "sample sizes must be non-negative and accepted must be positive"
        )

    seeds = {
        seed.seed_id: seed
        for seed in (
            TrajectorySeed.model_validate(row)
            for row in _read_jsonl(args.seed_file)
        )
    }
    accepted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for row in _read_jsonl(args.raw):
        seed_id = str(row["seed_id"])
        seed = seeds.get(seed_id)
        if seed is None:
            rejected.append(
                {"seed_id": seed_id, "automated_reasons": ["unknown_seed"]}
            )
            continue
        trajectory, reasons = parse_and_filter_trajectory(
            str(row["text"]), seed
        )
        if trajectory is None:
            rejected.append(
                {
                    "seed_id": seed_id,
                    "source": _source(seed_id),
                    "query": seed.query,
                    "expected_answer": seed.expected_answer,
                    "automated_reasons": reasons,
                }
            )
            continue
        accepted_by_source[_source(seed_id)].append(
            {
                "seed_id": seed_id,
                "source": _source(seed_id),
                "query": seed.query,
                "expected_answer": seed.expected_answer,
                "evidence": seed.evidence,
                "teacher_trajectory": trajectory.model_dump(mode="json"),
                "review": {
                    "schema_correct": None,
                    "evidence_faithful": None,
                    "answer_correct": None,
                    "notes": "",
                },
            }
        )

    review_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for source, rows in sorted(accepted_by_source.items()):
        selected = sorted(
            rows,
            key=lambda row: _sample_key(
                str(row["seed_id"]), args.sample_seed
            ),
        )[: args.accepted_per_source]
        review_rows.extend(selected)
        source_counts[source] = len(selected)
    rejected_rows = sorted(
        rejected,
        key=lambda row: _sample_key(
            str(row["seed_id"]), args.sample_seed
        ),
    )[: args.rejected_limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "accepted-review.jsonl"
    rejected_path = args.output_dir / "rejected-review.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rejected_rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "review_version": "trajectory-500-review-v1",
        "status": "awaiting_human_review",
        "accepted_source_counts": source_counts,
        "rejected_records": len(rejected_rows),
        "sample_seed": args.sample_seed,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (review_path, rejected_path)
        },
        "truth_boundary": (
            "This packet is deterministically sampled and has not been "
            "human-reviewed. Null review fields are not approvals."
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
