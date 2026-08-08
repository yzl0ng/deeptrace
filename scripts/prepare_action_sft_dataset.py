from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.agentic.tool_loop import ACTION_SYSTEM_PROMPT
from app.agentic.trajectory import AgentSFTResponse


def convert_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if [str(item["role"]) for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("source row must contain system/user/assistant")
    trajectory = AgentSFTResponse.model_validate_json(
        str(messages[2]["content"])
    )
    converted = [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": str(messages[1]["content"])},
    ]
    for step in trajectory.steps:
        action = {
            "rationale_summary": step.rationale_summary,
            "action": step.action,
            "arguments": step.arguments,
            "evidence_ids": step.evidence_ids,
            "final_answer": (
                trajectory.final_answer if step.action == "answer" else None
            ),
        }
        converted.append(
            {
                "role": "assistant",
                "content": json.dumps(action, ensure_ascii=False),
            }
        )
        if step.action != "answer":
            converted.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "status": "succeeded",
                            "observation": step.observation,
                            "evidence_ids": step.evidence_ids,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert full-trajectory SFT into action/tool turns."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()

    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("train.parquet", "validation.parquet"):
        source = args.input_dir / name
        frame = pd.read_parquet(source)
        frame["messages"] = [
            convert_messages(messages)
            for messages in frame["messages"].tolist()
        ]
        output = args.output_dir / name
        frame.to_parquet(output, index=False)
        artifacts[name] = {
            "records": len(frame),
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "mean_messages": sum(
                len(messages) for messages in frame["messages"]
            )
            / len(frame),
        }
    manifest = {
        "dataset_version": args.dataset_version,
        "source_dataset": args.input_dir.as_posix(),
        "format": "assistant_action_then_tool_observation",
        "artifacts": artifacts,
        "truth_boundary": (
            "The conversion separates existing Teacher actions and "
            "observations into turns. It does not independently verify every "
            "Teacher observation against source documents."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
