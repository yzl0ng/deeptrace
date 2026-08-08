from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic veRL train/validation Parquet files."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-records", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--dataset-version", default="sft-pilot-v1")
    parser.add_argument(
        "--forbidden-questions",
        type=Path,
        help="Optional JSONL whose question/query values must not overlap.",
    )
    args = parser.parse_args()
    import pandas as pd

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 < args.validation_records < len(rows):
        raise ValueError("validation_records must be between 1 and len(rows)-1")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("SFT record IDs must be unique")
    if len({row["trajectory_sha256"] for row in rows}) != len(rows):
        raise ValueError("SFT trajectory hashes must be unique")
    forbidden_overlap: list[str] = []
    if args.forbidden_questions is not None:
        forbidden = {
            _normalize_question(str(row.get("question") or row.get("query")))
            for row in _read_jsonl(args.forbidden_questions)
        }
        forbidden_overlap = sorted(
            {
                row["id"]
                for row in rows
                if _normalize_question(str(row["messages"][1]["content"]))
                in forbidden
            }
        )
        if forbidden_overlap:
            raise ValueError(
                "training records overlap forbidden questions: "
                + ", ".join(forbidden_overlap)
            )
    random.Random(args.seed).shuffle(rows)
    val = rows[: args.validation_records]
    train = rows[args.validation_records :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train.parquet": train,
        "validation.parquet": val,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, split in paths.items():
        path = args.output_dir / name
        pd.DataFrame(
            {
                "id": [row["id"] for row in split],
                "messages": [row["messages"] for row in split],
                "enable_thinking": [False] * len(split),
            }
        ).to_parquet(path, index=False)
        artifacts[name] = {
            "records": len(split),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "dataset_version": args.dataset_version,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed,
        "unique_ids": len({row["id"] for row in rows}),
        "unique_trajectory_hashes": len(
            {row["trajectory_sha256"] for row in rows}
        ),
        "forbidden_question_overlap": forbidden_overlap,
        "artifacts": artifacts,
        "truth_boundary": (
            "The split is deterministic and structurally validated. Automatic "
            "Teacher filtering is not human semantic review."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


if __name__ == "__main__":
    raise SystemExit(main())
