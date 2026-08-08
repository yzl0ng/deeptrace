from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.agentic.trajectory import (
    AGENT_SFT_SYSTEM_PROMPT,
    parse_sft_response,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    rewritten = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if [message["role"] for message in rewritten] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("each SFT row must contain system/user/assistant")
    rewritten[0]["content"] = AGENT_SFT_SYSTEM_PROMPT
    if parse_sft_response(rewritten[2]["content"]) is None:
        raise ValueError("assistant target does not match the SFT schema")
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite an SFT dataset to the explicit JSON contract."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()

    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    all_ids: list[str] = []
    for name in ("train.parquet", "validation.parquet"):
        source = args.input_dir / name
        frame = pd.read_parquet(source)
        frame["messages"] = [
            _rewrite_messages(messages)
            for messages in frame["messages"].tolist()
        ]
        output = args.output_dir / name
        frame.to_parquet(output, index=False)
        ids = [str(value) for value in frame["id"].tolist()]
        all_ids.extend(ids)
        artifacts[name] = {
            "records": len(frame),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        }
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("SFT record IDs must be unique across splits")

    contract_sha256 = hashlib.sha256(
        AGENT_SFT_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    manifest = {
        "dataset_version": args.dataset_version,
        "source_dataset": args.input_dir.as_posix(),
        "source_manifest_sha256": _sha256(
            args.input_dir / "manifest.json"
        ),
        "prompt_contract": {
            "version": "deeptrace-agent-sft-json-v1",
            "sha256": contract_sha256,
            "requires_exact_json_object": True,
        },
        "unique_ids": len(set(all_ids)),
        "artifacts": artifacts,
        "truth_boundary": (
            "Assistant targets and split membership are unchanged. Only the "
            "system prompt was rewritten to state the output contract."
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
