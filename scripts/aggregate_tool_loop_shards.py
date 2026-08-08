from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.agentic.tool_loop import ToolLoopResult, summarize_tool_loop_results


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate completed tool-loop result shards."
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_rows = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = [str(row["case_id"]) for row in expected_rows]
    expected_order = {
        case_id: index for index, case_id in enumerate(expected_ids)
    }
    if len(set(expected_ids)) != len(expected_ids):
        parser.error("case IDs are not unique")

    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for directory in args.result_dir:
        summary_path = directory / "summary.json"
        predictions_path = directory / "predictions.jsonl"
        if not summary_path.is_file() or not predictions_path.is_file():
            parser.error(f"incomplete result shard: {directory}")
        shard_summaries.append(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        rows.extend(
            json.loads(line)
            for line in predictions_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )

    reference = shard_summaries[0]
    config_keys = (
        "evaluation_version",
        "max_steps",
        "max_action_tokens",
        "schema_retries",
        "min_read_evidence",
        "require_evaluate_before_answer",
        "adaptive_soft_evidence_gate",
        "complexity_aware_soft_evidence_gate",
        "evidence_sufficiency_soft_gate",
        "evaluated_modes",
    )
    for summary in shard_summaries[1:]:
        for key in config_keys:
            if summary.get(key) != reference.get(key):
                parser.error(f"shard configuration differs at {key}")

    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["case_id"]))
        if key in keyed:
            parser.error(f"duplicate prediction: {key}")
        keyed[key] = row
    modes = [str(item) for item in reference["evaluated_modes"]]
    for mode in modes:
        observed = {
            case_id for candidate_mode, case_id in keyed
            if candidate_mode == mode
        }
        if observed != set(expected_ids):
            parser.error(
                f"{mode} case coverage differs: "
                f"missing={len(set(expected_ids) - observed)}, "
                f"extra={len(observed - set(expected_ids))}"
            )

    ordered_rows = sorted(
        keyed.values(),
        key=lambda row: (
            modes.index(str(row["mode"])),
            expected_order[str(row["case_id"])],
        ),
    )
    summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        results = [
            ToolLoopResult.model_validate(row)
            for row in ordered_rows
            if row["mode"] == mode
        ]
        summaries[mode] = summarize_tool_loop_results(results)

    comparison_gate = None
    if {"base", "sft"}.issubset(summaries):
        base_exact = summaries["base"]["metrics"]["answer_exact_rate"]
        sft_exact = summaries["sft"]["metrics"]["answer_exact_rate"]
        checks = {
            "sft_tool_loop_gate_passed": (
                summaries["sft"]["quality_gate"]["passed"]
            ),
            "sft_answer_exact_improves_by_5_points": (
                sft_exact >= base_exact + 0.05
            ),
        }
        comparison_gate = {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_rl"
                if all(checks.values())
                else "hold_before_rl"
            ),
        }

    summary = {
        **{
            key: reference.get(key)
            for key in config_keys
            if key in reference
        },
        "status": "succeeded",
        "cases": len(expected_ids),
        "physical_devices": [
            item["physical_device"] for item in shard_summaries
        ],
        "shards": len(shard_summaries),
        "modes": summaries,
        "comparison_gate": comparison_gate,
        "truth_boundary": reference["truth_boundary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in ordered_rows
        ),
        encoding="utf-8",
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "aggregation_version": "tool-loop-shard-aggregation-v1",
        "result_dirs": [str(item) for item in args.result_dir],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (predictions_path, summary_path)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
