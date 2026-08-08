from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agentic.sanitization import sanitize_text


@dataclass(frozen=True)
class NcclCheck:
    name: str
    required_devices: int
    topology_scope: str

    @property
    def world_size(self) -> int:
        return self.required_devices


NCCL_MATRIX: tuple[NcclCheck, ...] = (
    NcclCheck("single_gpu", 1, "one dynamically selected idle device"),
    NcclCheck("four_gpu", 4, "four dynamically selected idle devices"),
    NcclCheck("eight_gpu", 8, "all eight devices when simultaneously idle"),
)


def build_torchrun_command(
    check: NcclCheck,
    *,
    python_executable: str,
    result_path: Path,
    timeout_seconds: int,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={check.world_size}",
        "-m",
        "app.agentic.distributed_smoke",
        "--output",
        result_path.as_posix(),
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def run_nccl_matrix(
    *,
    selected_names: list[str],
    python_executable: str,
    output_dir: Path,
    timeout_seconds: int,
    execute: bool,
    max_used_memory_mib: int = 64,
) -> dict[str, Any]:
    known = {check.name: check for check in NCCL_MATRIX}
    unknown = sorted(set(selected_names) - set(known))
    if unknown:
        raise ValueError(f"unknown NCCL checks: {', '.join(unknown)}")
    if max_used_memory_mib < 0:
        raise ValueError("max_used_memory_mib must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in selected_names:
        check = known[name]
        result_path = output_dir / f"{check.name}.json"
        command = build_torchrun_command(
            check,
            python_executable=python_executable,
            result_path=result_path,
            timeout_seconds=timeout_seconds,
        )
        row: dict[str, Any] = {
            "name": check.name,
            "cuda_visible_devices": "selected at execution",
            "world_size": check.world_size,
            "topology_scope": check.topology_scope,
            "status": "planned",
            "result_path": result_path.name,
            "command": [sanitize_text(part) for part in command],
        }
        if execute:
            idle_devices = discover_idle_devices(
                max_used_memory_mib=max_used_memory_mib
            )
            selected_devices = idle_devices[: check.world_size]
            row["idle_devices_at_preflight"] = idle_devices
            if len(selected_devices) < check.world_size:
                row["status"] = "pending"
                row["reason"] = (
                    f"requires {check.world_size} idle GPUs but preflight "
                    f"found {len(idle_devices)}"
                )
                rows.append(row)
                continue
            row["cuda_visible_devices"] = ",".join(
                map(str, selected_devices)
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = row["cuda_visible_devices"]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds + 30,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                row["status"] = "failed"
                row["return_code"] = None
                row["stdout"] = sanitize_text(exc.stdout or "")
                row["stderr"] = "torchrun timed out"
            else:
                row["return_code"] = completed.returncode
                row["stdout"] = sanitize_text(completed.stdout.strip())
                row["stderr"] = sanitize_text(completed.stderr.strip())
                if completed.returncode == 0 and result_path.is_file():
                    payload = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    row["status"] = (
                        "succeeded"
                        if payload.get("status") == "succeeded"
                        and payload.get("world_size") == check.world_size
                        and payload.get("observed_sum")
                        == payload.get("expected_sum")
                        else "failed"
                    )
                else:
                    row["status"] = "failed"
        rows.append(row)
        if execute and row["status"] != "succeeded":
            break

    return {
        "validation_version": "nccl-smoke-v4",
        # The remote training runtime is Python 3.10 and has no datetime.UTC.
        "observed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "status": _matrix_status(rows, execute=execute),
        "selection_policy": (
            "At each launch, select the lowest-index GPUs with no reported "
            "compute process and memory usage at or below "
            f"{max_used_memory_mib} MiB."
        ),
        "planned_checks": rows,
        "truth_boundary": (
            "A planned or pending result launches nothing for that row. A "
            "succeeded result covers only the dynamically selected devices "
            "recorded by that row. The preflight reduces interference risk "
            "but cannot eliminate the race with another process starting."
        ),
    }


def discover_idle_devices(
    *,
    max_used_memory_mib: int = 64,
    require_no_compute_processes: bool = True,
) -> list[int]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    busy_uuids = {
        line.strip()
        for line in processes.stdout.splitlines()
        if line.strip()
    }
    candidates: list[int] = []
    for line in inventory.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        index_text, gpu_uuid, used_text = parts
        process_ok = (
            not require_no_compute_processes or gpu_uuid not in busy_uuids
        )
        if process_ok and int(used_text) <= max_used_memory_mib:
            candidates.append(int(index_text))
    return sorted(candidates)


def _matrix_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not execute:
        return "planned"
    statuses = {row["status"] for row in rows}
    if statuses == {"succeeded"}:
        return "succeeded"
    if "failed" in statuses:
        return "failed"
    if "succeeded" in statuses:
        return "partial"
    return "pending"


def write_nccl_matrix(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    (output_dir / "nccl-validation.json").write_bytes(validation)
    artifacts = {
        "nccl-validation.json": hashlib.sha256(validation).hexdigest()
    }
    for row in payload["planned_checks"]:
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
