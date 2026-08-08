from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.agentic.trajectory import (
    TrajectorySeed,
    build_pilot_seeds,
    parse_and_filter_trajectory,
    teacher_messages,
    to_sft_record,
)
from app.core.llm import DeepSeekClient, DeepSeekSettings, LLMError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and filter the Phase 5 DeepSeek trajectory pilot."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed-file",
        type=Path,
        help="Optional JSONL file of TrajectorySeed records.",
    )
    parser.add_argument(
        "--trajectory-version",
        default="trajectory-pilot-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/trajectory-pilot-v1"),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 5000:
        raise ValueError("limit must be between 1 and 5000")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    root = Path(__file__).resolve().parents[1]
    if args.seed_file is None:
        if args.limit > 100:
            raise ValueError("--seed-file is required when limit exceeds 100")
        seeds = build_pilot_seeds(root, target=100)[: args.limit]
    else:
        seeds = [
            TrajectorySeed.model_validate(row)
            for row in _read_jsonl(args.seed_file)
        ][: args.limit]
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise ValueError("seed IDs must be unique")
    settings = DeepSeekSettings.from_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": args.output_dir / "raw.jsonl",
        "accepted": args.output_dir / "sft.jsonl",
        "rejected": args.output_dir / "rejected.jsonl",
    }
    existing = {
        name: _read_jsonl(path) if path.is_file() else []
        for name, path in paths.items()
    }
    processed_ids = {
        str(row["seed_id"])
        for row in existing["rejected"]
    } | {
        str(row["id"])
        for row in existing["accepted"]
    }
    remaining = [seed for seed in seeds if seed.seed_id not in processed_ids]
    for path in paths.values():
        path.touch(exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            lambda seed: _generate_one(seed, settings),
            remaining,
        )
        with (
            paths["raw"].open("a", encoding="utf-8") as raw_stream,
            paths["accepted"].open("a", encoding="utf-8") as accepted_stream,
            paths["rejected"].open("a", encoding="utf-8") as rejected_stream,
        ):
            for index, (raw_row, accepted_row, rejected_row) in enumerate(
                results,
                start=len(processed_ids) + 1,
            ):
                if raw_row:
                    _append_jsonl(raw_stream, raw_row)
                if accepted_row:
                    _append_jsonl(accepted_stream, accepted_row)
                if rejected_row:
                    _append_jsonl(rejected_stream, rejected_row)
                print(
                    f"[{index}/{len(seeds)}] "
                    f"accepted={accepted_row is not None}",
                    flush=True,
                )

    raw = _read_jsonl(paths["raw"])
    accepted = _read_jsonl(paths["accepted"])
    rejected = _read_jsonl(paths["rejected"])
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths.values()
    }
    usage = {
        key: sum(
            int(row.get("usage", {}).get(key, 0))
            for row in raw
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    summary = {
        "trajectory_version": args.trajectory_version,
        "requested": len(seeds),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(seeds),
        "usage": usage,
        "artifacts": hashes,
        "truth_boundary": (
            "Teacher outputs are automatically schema-filtered, not human "
            "quality approved. Raw outputs may contain model errors."
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def _generate_one(
    seed: TrajectorySeed,
    settings: DeepSeekSettings,
) -> tuple[
    dict[str, object] | None,
    dict[str, Any] | None,
    dict[str, object] | None,
]:
    error: LLMError | None = None
    for attempt in range(3):
        try:
            result = DeepSeekClient(settings).generate_messages(
                teacher_messages(seed)
            )
            break
        except LLMError as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        return (
            None,
            None,
            {
                "seed_id": seed.seed_id,
                "reasons": [f"request_error:{error.code}"],
            },
        )
    raw_row: dict[str, object] = {
        "seed_id": seed.seed_id,
        "model": result.model,
        "text": result.text,
        "usage": asdict(result.usage),
    }
    trajectory, reasons = parse_and_filter_trajectory(result.text, seed)
    if trajectory is None:
        return raw_row, None, {"seed_id": seed.seed_id, "reasons": reasons}
    return raw_row, to_sft_record(trajectory), None


if __name__ == "__main__":
    raise SystemExit(main())
