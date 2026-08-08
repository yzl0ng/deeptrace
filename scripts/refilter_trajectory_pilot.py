from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.agentic.trajectory import (
    TrajectorySeed,
    build_pilot_seeds,
    parse_and_filter_trajectory,
    to_sft_record,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refilter saved raw Teacher outputs without new API calls."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument(
        "--trajectory-version",
        default="trajectory-pilot-v1-refiltered",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.seed_file is None:
        seed_rows = build_pilot_seeds(root)
    else:
        seed_rows = [
            TrajectorySeed.model_validate(row)
            for row in _read_jsonl(args.seed_file)
        ]
    seeds = {seed.seed_id: seed for seed in seed_rows}
    raw_rows = [
        json.loads(line)
        for line in args.raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in raw_rows:
        seed = seeds.get(str(row["seed_id"]))
        if seed is None:
            rejected.append(
                {"seed_id": row["seed_id"], "reasons": ["unknown_seed"]}
            )
            continue
        trajectory, reasons = parse_and_filter_trajectory(
            str(row["text"]),
            seed,
        )
        if trajectory is None:
            rejected.append({"seed_id": seed.seed_id, "reasons": reasons})
        else:
            accepted.append(to_sft_record(trajectory))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "sft.jsonl": _jsonl(accepted),
        "rejected.jsonl": _jsonl(rejected),
    }
    hashes: dict[str, str] = {}
    for name, content in payloads.items():
        encoded = content.encode()
        (args.output_dir / name).write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    summary = {
        "trajectory_version": args.trajectory_version,
        "raw_reference": args.raw.as_posix(),
        "requested": len(raw_rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(raw_rows),
        "unique_accepted_ids": len({row["id"] for row in accepted}),
        "unique_trajectory_hashes": len(
            {row["trajectory_sha256"] for row in accepted}
        ),
        "artifacts": hashes,
        "truth_boundary": (
            "Outputs passed deterministic schema and evidence-ID checks. "
            "They have not received human semantic review."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if accepted else 2


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
