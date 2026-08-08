from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_cases(
    validation_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seeds = {str(row["seed_id"]): row for row in seed_rows}
    cases: list[dict[str, Any]] = []
    for row in validation_rows:
        case_id = str(row["id"])
        seed = seeds.get(case_id)
        if seed is None:
            raise ValueError(f"missing seed for validation ID {case_id}")
        expected_answer = str(seed.get("expected_answer", ""))
        raw_evidence = list(seed.get("evidence", []))
        evidence = [
            {
                "evidence_id": str(item["evidence_id"]),
                "title": str(item.get("title", "")),
                "content": str(item.get("content") or item.get("text") or ""),
                "source": str(
                    item.get("source") or f"frozen-seed:{case_id}"
                ),
            }
            for item in raw_evidence
        ]
        if not expected_answer or not evidence:
            raise ValueError(
                f"tool-loop case {case_id} needs answer and evidence"
            )
        cases.append(
            {
                "case_id": case_id,
                "dataset": case_id.split("-", 1)[0],
                "split": "dev",
                "language": str(seed.get("language", "en")),
                "question": str(seed["query"]),
                "expected_answer": expected_answer,
                "answer_aliases": [],
                "evidence": evidence,
                "gold_claims": [
                    {
                        "claim_id": f"{case_id}-gold-answer",
                        "text": (
                            f'The answer to "{seed["query"]}" is '
                            f'"{expected_answer}".'
                        ),
                        "supporting_evidence_ids": [
                            str(item["evidence_id"]) for item in evidence
                        ],
                        "contradicting_evidence_ids": [],
                    }
                ],
                "reviewed": False,
                "source_record_id": case_id,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join SFT validation IDs to frozen evidence for tool-loop dev."
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    validation_rows = pd.read_parquet(args.validation).to_dict("records")
    seed_rows = [
        json.loads(line)
        for line in args.seeds.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = build_cases(validation_rows, seed_rows)
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in cases
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "dataset_version": "sft-tool-loop-dev-v1",
        "cases": len(cases),
        "reviewed": False,
        "split": "dev",
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "truth_boundary": (
            "This is a tuning-only tool-loop dev set joined to frozen seed "
            "evidence. It is not an unseen final test."
        ),
    }
    (args.output.parent / "tool-loop-dev-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
