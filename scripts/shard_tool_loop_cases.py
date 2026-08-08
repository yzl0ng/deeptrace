from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically shard tool-loop cases round-robin."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 2:
        parser.error("shards must be at least two")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error("input contains no cases")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, object]] = {}
    all_ids: list[str] = []
    for shard_index in range(args.shards):
        shard_rows = rows[shard_index :: args.shards]
        payload = "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in shard_rows
        ).encode("utf-8")
        path = args.output_dir / f"shard-{shard_index}.jsonl"
        path.write_bytes(payload)
        ids = [str(row["case_id"]) for row in shard_rows]
        all_ids.extend(ids)
        datasets: dict[str, int] = {}
        for row in shard_rows:
            name = str(row["dataset"])
            datasets[name] = datasets.get(name, 0) + 1
        artifacts[path.name] = {
            "cases": len(shard_rows),
            "datasets": dict(sorted(datasets.items())),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    input_ids = [str(row["case_id"]) for row in rows]
    if sorted(all_ids) != sorted(input_ids):
        raise RuntimeError("shards do not exactly cover input case IDs")
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("shards contain duplicate case IDs")
    manifest = {
        "shard_version": "tool-loop-round-robin-v1",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "cases": len(rows),
        "shards": args.shards,
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
