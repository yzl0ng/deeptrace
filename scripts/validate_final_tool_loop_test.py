from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.agentic.tool_loop import ToolLoopEvalCase


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_questions(rows: list[dict[str, Any]]) -> set[str]:
    questions: set[str] = set()
    for row in rows:
        value = str(row.get("question") or row.get("query") or "")
        if value.strip():
            questions.add(" ".join(value.casefold().split()))
    return questions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen final tool-loop test invariants."
    )
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--distractor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controlled_raw = _read_jsonl(args.controlled)
    distractor_raw = _read_jsonl(args.distractor)
    controlled = [
        ToolLoopEvalCase.model_validate(row) for row in controlled_raw
    ]
    distractor = [
        ToolLoopEvalCase.model_validate(row) for row in distractor_raw
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if len(controlled) != 90 or len(distractor) != 90:
        errors.append("both final test variants must contain 90 cases")
    controlled_by_id = {item.case_id: item for item in controlled}
    distractor_by_id = {item.case_id: item for item in distractor}
    if len(controlled_by_id) != len(controlled):
        errors.append("controlled case IDs are not unique")
    if len(distractor_by_id) != len(distractor):
        errors.append("distractor case IDs are not unique")
    if set(controlled_by_id) != set(distractor_by_id):
        errors.append("variant case IDs differ")

    for case_id, base in controlled_by_id.items():
        extended = distractor_by_id.get(case_id)
        if extended is None:
            continue
        if (
            base.question,
            base.expected_answer,
            base.answer_aliases,
            base.gold_claims,
        ) != (
            extended.question,
            extended.expected_answer,
            extended.answer_aliases,
            extended.gold_claims,
        ):
            errors.append(f"{case_id}: labels differ across variants")
        gold_ids = {
            evidence_id
            for claim in base.gold_claims
            for evidence_id in claim.supporting_evidence_ids
        }
        controlled_ids = {
            item.evidence_id for item in base.evidence
        }
        distractor_ids = {
            item.evidence_id for item in extended.evidence
        }
        if controlled_ids != gold_ids:
            errors.append(
                f"{case_id}: controlled evidence is not exactly gold"
            )
        if not gold_ids.issubset(distractor_ids):
            errors.append(
                f"{case_id}: distractor evidence omits a gold ID"
            )
        if len(distractor_ids) < len(controlled_ids):
            errors.append(
                f"{case_id}: distractor variant has less evidence"
            )

    final_questions = _normalized_questions(controlled_raw)
    overlap_details: dict[str, int] = {}
    for path in args.exclude_jsonl:
        overlap = final_questions & _normalized_questions(_read_jsonl(path))
        overlap_details[str(path)] = len(overlap)
        if overlap:
            errors.append(f"{path}: {len(overlap)} overlapping questions")

    expected_artifacts = manifest["artifacts"]
    hashes = {
        args.controlled.name: _sha256(args.controlled),
        args.distractor.name: _sha256(args.distractor),
    }
    for name, digest in hashes.items():
        if digest != expected_artifacts[name]["sha256"]:
            errors.append(f"{name}: SHA-256 does not match manifest")

    dataset_counts = Counter(item.dataset for item in controlled)
    if dataset_counts != {
        "hotpotqa": 30,
        "2wikimultihopqa": 30,
        "musique": 30,
    }:
        errors.append(f"dataset balance is wrong: {dataset_counts}")

    report = {
        "validation_version": "final-tool-loop-test-validation-v1",
        "passed": not errors,
        "cases": len(controlled),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "controlled_sha256": hashes[args.controlled.name],
        "distractor_sha256": hashes[args.distractor.name],
        "excluded_question_overlaps": overlap_details,
        "errors": errors,
        "truth_boundary": (
            "This validates schema, balance, hashes, variant alignment, and "
            "question-level overlap. It does not manually verify every label."
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
