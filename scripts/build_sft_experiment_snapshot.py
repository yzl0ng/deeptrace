from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAIN_STEP = re.compile(
    r"step:(?P<step>\d+)\s+-\s+train/loss:(?P<loss>[0-9.eE+-]+)"
)
VAL_STEP = re.compile(
    r"step:(?P<step>\d+)\s+-\s+val/loss:(?P<loss>[0-9.eE+-]+)"
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _training_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "not_started"}
    fields = path.read_text(encoding="utf-8").strip().split("\t")
    return {
        "updated_at": fields[0] if fields else None,
        "state": fields[1] if len(fields) > 1 else "unknown",
        "details": fields[2:],
    }


def _training_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    train = [
        {"step": int(match["step"]), "loss": float(match["loss"])}
        for match in TRAIN_STEP.finditer(text)
    ]
    validation = [
        {"step": int(match["step"]), "loss": float(match["loss"])}
        for match in VAL_STEP.finditer(text)
    ]
    selected = None
    for line in text.splitlines():
        if line.startswith("SELECTED_GPUS="):
            selected = line.split("=", 1)[1].strip()
            break
    return {
        "selected_gpus": selected,
        "observed_train_steps": len(train),
        "latest_step": train[-1]["step"] if train else 0,
        "latest_train_loss": train[-1]["loss"] if train else None,
        "latest_validation_loss": (
            validation[-1]["loss"] if validation else None
        ),
        "loss_curve": train,
        "validation_curve": validation,
    }


def _checkpoint_status(root: Path) -> dict[str, Any]:
    checkpoints = sorted(
        (
            path
            for path in root.glob("global_step_*")
            if path.is_dir()
        ),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        return {"state": "pending", "latest_step": None}
    latest = checkpoints[-1]
    components = {
        "model_shards": sorted(
            path.name for path in latest.glob("model_world_size_*.pt")
        ),
        "optimizer_shards": sorted(
            path.name for path in latest.glob("optim_world_size_*.pt")
        ),
        "extra_state_shards": sorted(
            path.name for path in latest.glob("extra_state_world_size_*.pt")
        ),
        "dataloader_state": (latest / "data.pt").is_file(),
        "huggingface_config": (latest / "huggingface").is_dir(),
    }
    complete = (
        len(components["model_shards"]) == 2
        and len(components["optimizer_shards"]) == 2
        and len(components["extra_state_shards"]) == 2
        and components["dataloader_state"]
        and components["huggingface_config"]
    )
    return {
        "state": "complete" if complete else "incomplete",
        "latest_step": int(latest.name.rsplit("_", 1)[-1]),
        "path": latest.as_posix(),
        "components": components,
    }


def build_snapshot(
    *,
    trajectory_audit: dict[str, Any] | None,
    dataset_manifest: dict[str, Any] | None,
    dataset_validation: dict[str, Any] | None,
    training_status: dict[str, Any],
    training_metrics: dict[str, Any],
    checkpoint: dict[str, Any],
    adapter_manifest: dict[str, Any] | None,
    adapter_validation: dict[str, Any] | None,
    dev_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "snapshot_version": "sft-experiment-snapshot-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_quality": trajectory_audit,
        "dataset": {
            "manifest": dataset_manifest,
            "validation": dataset_validation,
        },
        "training": {
            "expected_epoch_steps": 210,
            "status": training_status,
            "metrics": training_metrics,
        },
        "checkpoint": checkpoint,
        "adapter": {
            "manifest": adapter_manifest,
            "load_validation": adapter_validation,
        },
        "dev_evaluation": dev_evaluation,
        "truth_boundary": (
            "Missing downstream artifacts remain null or pending. A completed "
            "training process is not a quality improvement claim; Base/SFT "
            "quality is reported only by the independent dev evaluation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a truthful UI-ready SFT experiment snapshot."
    )
    parser.add_argument("--trajectory-audit", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--dev-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = build_snapshot(
        trajectory_audit=_read_json(args.trajectory_audit),
        dataset_manifest=_read_json(args.dataset_manifest),
        dataset_validation=_read_json(args.dataset_validation),
        training_status=_training_status(args.status),
        training_metrics=_training_metrics(args.train_log),
        checkpoint=_checkpoint_status(args.checkpoint_root),
        adapter_manifest=_read_json(args.adapter_dir / "manifest.json"),
        adapter_validation=_read_json(
            args.adapter_dir / "load-validation.json"
        ),
        dev_evaluation=_read_json(args.dev_evaluation),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote experiment snapshot with state "
        f"{snapshot['training']['status']['state']} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
