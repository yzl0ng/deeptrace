from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def assess(
    summary: dict[str, Any],
    *,
    answer_target: float,
) -> dict[str, Any]:
    sft = summary["modes"]["sft"]["metrics"]
    checks = {
        "answer_exact_target": sft["answer_exact_rate"] >= answer_target,
        "completion_at_least_95_percent": sft["completion_rate"] >= 0.95,
        "no_unknown_evidence_ids": (
            sft["unknown_evidence_id_attempts"] == 0
        ),
        "no_invalid_actions": sft["invalid_action_attempts"] == 0,
        "supporting_evidence_recall_at_least_90_percent": (
            sft["mean_supporting_evidence_recall"] >= 0.90
        ),
    }
    answer_passed = checks["answer_exact_target"]
    production_passed = all(checks.values())
    if production_passed:
        decision = "production_quality_gate_passed"
    elif answer_passed:
        decision = "answer_target_passed_protocol_gate_failed"
    else:
        decision = "answer_target_failed_continue_training"
    return {
        "assessment_version": "tool-loop-target-assessment-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answer_target": answer_target,
        "observed": sft,
        "checks": checks,
        "answer_target_passed": answer_passed,
        "production_quality_gate_passed": production_passed,
        "decision": decision,
        "truth_boundary": (
            "This decision uses tuning-only dev metrics. Passing it does not "
            "replace an independent unseen final test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for and assess a tool-loop evaluation summary."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-target", type=float, default=0.50)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 0 <= args.answer_target <= 1:
        parser.error("answer target must be between 0 and 1")

    deadline = time.monotonic() + args.wait_timeout_seconds
    while not args.summary.is_file():
        if args.wait_timeout_seconds <= 0 or time.monotonic() >= deadline:
            parser.error(f"summary not found: {args.summary}")
        time.sleep(max(args.poll_seconds, 1))

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = assess(summary, answer_target=args.answer_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
