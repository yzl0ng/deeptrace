from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agentic.sanitization import sanitize_text

SMOKE_MODES = ("cuda_extensions", "vllm_inference")


def build_runtime_smoke_command(
    *,
    mode: str,
    python_executable: str,
    model_path: str,
    result_path: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[str]:
    if mode not in SMOKE_MODES:
        raise ValueError(f"unknown runtime smoke mode: {mode}")
    command = [
        python_executable,
        "-m",
        "app.agentic.gpu_runtime_smoke",
        "--mode",
        mode,
        "--output",
        result_path.as_posix(),
    ]
    if mode == "vllm_inference":
        command.extend(
            [
                "--model-path",
                model_path,
                "--max-model-len",
                str(max_model_len),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
            ]
        )
    return command


def run_runtime_smoke_plan(
    *,
    python_executable: str,
    model_path: str,
    output_dir: Path,
    device: int,
    timeout_seconds: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    execute: bool,
    reservation_confirmed: bool,
    reservation_reference: str,
) -> dict[str, Any]:
    if device < 0:
        raise ValueError("device must be non-negative")
    if execute and (
        not reservation_confirmed or not reservation_reference.strip()
    ):
        raise ValueError(
            "--execute requires both --reservation-confirmed and a non-empty "
            "--reservation-reference"
        )
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for mode in SMOKE_MODES:
        result_path = output_dir / f"{mode}.json"
        command = build_runtime_smoke_command(
            mode=mode,
            python_executable=python_executable,
            model_path=model_path,
            result_path=result_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        row: dict[str, Any] = {
            "mode": mode,
            "device": device,
            "status": "planned",
            "result_path": result_path.name,
            "command": [sanitize_text(part) for part in command],
        }
        if execute:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                row["status"] = "failed"
                row["return_code"] = None
                row["stdout"] = sanitize_text(exc.stdout or "")
                row["stderr"] = "runtime smoke timed out"
            else:
                row["return_code"] = completed.returncode
                row["stdout"] = sanitize_text(completed.stdout.strip())
                row["stderr"] = sanitize_text(completed.stderr.strip())
                if completed.returncode == 0 and result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    row["status"] = (
                        "succeeded"
                        if result.get("status") == "succeeded"
                        and result.get("mode") == mode
                        else "failed"
                    )
                else:
                    row["status"] = "failed"
        rows.append(row)
        if execute and row["status"] != "succeeded":
            break

    return {
        "validation_version": "gpu-runtime-smoke-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "succeeded"
            if execute
            and len(rows) == len(SMOKE_MODES)
            and all(row["status"] == "succeeded" for row in rows)
            else "planned" if not execute else "failed"
        ),
        "reservation_confirmed": reservation_confirmed,
        "reservation_reference": reservation_reference,
        "device": device,
        "checks": rows,
        "truth_boundary": (
            "A planned result launches no GPU work. A succeeded result proves "
            "only the two recorded single-GPU checks inside the referenced "
            "reservation; it does not prove multi-GPU training readiness."
        ),
    }


def write_runtime_smoke_plan(
    payload: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    (output_dir / "runtime-validation.json").write_bytes(validation)
    artifacts = {
        "runtime-validation.json": hashlib.sha256(validation).hexdigest()
    }
    for row in payload["checks"]:
        result_path = output_dir / row["result_path"]
        if row["status"] != "planned" and result_path.is_file():
            artifacts[result_path.name] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
    manifest = {
        "validation_version": payload["validation_version"],
        "status": payload["status"],
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
