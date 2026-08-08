from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_dev_cases(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for row in rows:
        messages = list(row["messages"])
        user_messages = [
            message for message in messages if message["role"] == "user"
        ]
        assistant_messages = [
            message
            for message in messages
            if message["role"] == "assistant"
        ]
        if len(user_messages) != 1 or len(assistant_messages) != 1:
            raise ValueError(
                f"{row['id']} must have exactly one user and assistant turn"
            )
        target = json.loads(str(assistant_messages[0]["content"]))
        final_answer = str(target["final_answer"]).strip()
        if not final_answer:
            raise ValueError(f"{row['id']} has an empty final answer")
        cases.append(
            {
                "case_id": str(row["id"]),
                "question": str(user_messages[0]["content"]),
                "expected_answer": final_answer,
                "split": "validation_dev",
            }
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("dev case IDs must be unique")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the SFT validation parquet into a dev JSONL."
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    rows = pd.read_parquet(args.validation).to_dict(orient="records")
    cases = build_dev_cases(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} validation dev cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
