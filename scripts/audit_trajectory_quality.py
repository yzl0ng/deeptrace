from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.agentic.trajectory import (
    TrajectorySeed,
    parse_and_filter_trajectory,
)

ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source(seed_id: str) -> str:
    return seed_id.split("-train-", 1)[0]


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = ARTICLES.sub(" ", value)
    value = NON_WORD.sub(" ", value)
    return " ".join(value.split())


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum(
        (Counter(prediction_tokens) & Counter(reference_tokens)).values()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _contains_answer(text: str, answer: str) -> bool:
    normalized_answer = normalize_answer(answer)
    return bool(
        normalized_answer
        and normalized_answer in normalize_answer(text)
    )


def audit_records(
    raw_rows: list[dict[str, Any]],
    seeds: dict[str, TrajectorySeed],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rejected_reasons: Counter[str] = Counter()
    accepted_rows: list[dict[str, Any]] = []
    unknown_seeds = 0

    for raw in raw_rows:
        seed_id = str(raw.get("seed_id", ""))
        seed = seeds.get(seed_id)
        if seed is None:
            unknown_seeds += 1
            rejected_reasons["unknown_seed"] += 1
            continue
        trajectory, reasons = parse_and_filter_trajectory(
            str(raw.get("text", "")), seed
        )
        if trajectory is None:
            rejected_reasons.update(reasons)
            continue

        final_answer = trajectory.final_answer
        expected_answer = seed.expected_answer
        known_ids = {
            str(item["evidence_id"])
            for item in seed.evidence
            if item.get("evidence_id")
        }
        used_ids = {
            evidence_id
            for step in trajectory.steps
            for evidence_id in step.evidence_ids
        }
        evidence_text = " ".join(
            " ".join(
                str(item.get(field, ""))
                for field in ("title", "text")
            )
            for item in seed.evidence
        )
        actions = [step.action for step in trajectory.steps]
        accepted_rows.append(
            {
                "seed_id": seed_id,
                "source": _source(seed_id),
                "step_count": len(actions),
                "actions": actions,
                "normalized_exact": (
                    normalize_answer(final_answer)
                    == normalize_answer(expected_answer)
                ),
                "expected_in_final": _contains_answer(
                    final_answer, expected_answer
                ),
                "token_f1": token_f1(final_answer, expected_answer),
                "expected_supported_by_seed_evidence": _contains_answer(
                    evidence_text, expected_answer
                ),
                "final_supported_by_seed_evidence": _contains_answer(
                    evidence_text, final_answer
                ),
                "last_action_is_answer": actions[-1] == "answer",
                "answer_action_count": actions.count("answer"),
                "has_search": "search" in actions,
                "has_read_page": "read_page" in actions,
                "has_evaluate_evidence": "evaluate_evidence" in actions,
                "used_evidence_count": len(used_ids),
                "known_evidence_only": used_ids.issubset(known_ids),
            }
        )

    metrics = {
        "input_records": len(raw_rows),
        "known_seeds": len(seeds),
        "accepted_records": len(accepted_rows),
        "rejected_records": len(raw_rows) - len(accepted_rows),
        "unknown_seed_records": unknown_seeds,
        "acceptance_rate": _rate(len(accepted_rows), len(raw_rows)),
        "rejection_reasons": dict(sorted(rejected_reasons.items())),
        "accepted": _aggregate(accepted_rows),
        "by_source": {
            source: _aggregate(rows)
            for source, rows in sorted(
                _group_by_source(accepted_rows).items()
            )
        },
    }
    gate_checks = {
        "accepted_at_least_400_records": len(accepted_rows) >= 400,
        "usable_yield_at_least_90_percent": (
            metrics["acceptance_rate"] >= 0.90
        ),
        "normalized_exact_at_least_99_percent": (
            metrics["accepted"]["normalized_exact_rate"] >= 0.99
        ),
        "last_action_answer_at_least_99_percent": (
            metrics["accepted"]["last_action_is_answer_rate"] >= 0.99
        ),
        "exactly_one_answer_at_least_99_percent": (
            metrics["accepted"]["exactly_one_answer_rate"] >= 0.99
        ),
        "known_evidence_only": (
            metrics["accepted"]["known_evidence_only_rate"] == 1.0
        ),
    }
    metrics["quality_gate"] = {
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
        "note": (
            "Lexical evidence support is diagnostic and is not a gate: "
            "multi-hop support may be entailed without repeating the exact "
            "final answer string."
        ),
    }
    return metrics, accepted_rows


def _group_by_source(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    return grouped


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    step_counts = [int(row["step_count"]) for row in rows]
    action_counts = Counter(
        action for row in rows for action in row["actions"]
    )
    return {
        "records": count,
        "normalized_exact_rate": _rate(
            sum(bool(row["normalized_exact"]) for row in rows), count
        ),
        "expected_in_final_rate": _rate(
            sum(bool(row["expected_in_final"]) for row in rows), count
        ),
        "mean_token_f1": (
            statistics.fmean(float(row["token_f1"]) for row in rows)
            if rows
            else 0.0
        ),
        "expected_supported_by_seed_evidence_rate": _rate(
            sum(
                bool(row["expected_supported_by_seed_evidence"])
                for row in rows
            ),
            count,
        ),
        "final_supported_by_seed_evidence_rate": _rate(
            sum(
                bool(row["final_supported_by_seed_evidence"])
                for row in rows
            ),
            count,
        ),
        "last_action_is_answer_rate": _rate(
            sum(bool(row["last_action_is_answer"]) for row in rows), count
        ),
        "exactly_one_answer_rate": _rate(
            sum(int(row["answer_action_count"]) == 1 for row in rows), count
        ),
        "has_search_rate": _rate(
            sum(bool(row["has_search"]) for row in rows), count
        ),
        "has_read_page_rate": _rate(
            sum(bool(row["has_read_page"]) for row in rows), count
        ),
        "has_evaluate_evidence_rate": _rate(
            sum(bool(row["has_evaluate_evidence"]) for row in rows), count
        ),
        "known_evidence_only_rate": _rate(
            sum(bool(row["known_evidence_only"]) for row in rows), count
        ),
        "uses_evidence_rate": _rate(
            sum(int(row["used_evidence_count"]) > 0 for row in rows), count
        ),
        "step_count": {
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
            "mean": statistics.fmean(step_counts) if step_counts else 0.0,
            "median": statistics.median(step_counts) if step_counts else 0.0,
        },
        "action_counts": dict(sorted(action_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit accepted trajectory quality over a full raw run."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_rows = _read_jsonl(args.raw)
    seed_rows = _read_jsonl(args.seed_file)
    seeds = {
        seed.seed_id: seed
        for seed in (
            TrajectorySeed.model_validate(row) for row in seed_rows
        )
    }
    if len(seeds) != len(seed_rows):
        raise ValueError("seed IDs must be unique")

    metrics, accepted_rows = audit_records(raw_rows, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "accepted-quality.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in accepted_rows
        ),
        encoding="utf-8",
    )
    metrics["artifacts"] = {
        records_path.name: {
            "bytes": records_path.stat().st_size,
            "sha256": hashlib.sha256(
                records_path.read_bytes()
            ).hexdigest(),
        }
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["quality_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
