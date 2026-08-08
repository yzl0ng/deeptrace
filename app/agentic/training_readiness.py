from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

READINESS_VERSION = "training-readiness-v1"


class GateStatus(StrEnum):
    PASSED = "passed"
    BLOCKED = "blocked"
    PENDING = "pending"


class ReadinessGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: GateStatus
    evidence: str
    remediation: str | None = None


class Confirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool = False
    reference: str = ""


class RuntimeConfirmation(Confirmation):
    python_executable: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""


class CudaToolkitConfirmation(Confirmation):
    version: str = ""
    executable: str = ""
    compile_target: str = ""
    compiler_smoke_sha256: str = ""


class StorageConfirmation(Confirmation):
    project_root: str = ""
    model_root: str = ""
    data_root: str = ""
    checkpoint_root: str = ""


class StudentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = ""
    revision: str = ""
    local_path: str = ""
    reference: str = ""


class OperatorInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_reservation: Confirmation = Field(default_factory=Confirmation)
    runtime_environment: RuntimeConfirmation = Field(
        default_factory=RuntimeConfirmation
    )
    cuda_toolkit: CudaToolkitConfirmation = Field(
        default_factory=CudaToolkitConfirmation
    )
    storage: StorageConfirmation = Field(
        default_factory=StorageConfirmation
    )
    student: StudentSelection = Field(default_factory=StudentSelection)
    git_sync: Confirmation = Field(default_factory=Confirmation)


class TrainingStackRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    agent_r1_commit: str
    verl_commit: str
    verl_tag: str
    python_minimum: str
    cuda_minimum: str
    cuda_12_minor_compatibility_driver_minimum: str
    cuda_12_8_full_driver_minimum: str
    required_nccl_world_sizes: list[int]


class TrainingReadinessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readiness_version: str = READINESS_VERSION
    evaluated_at: str
    status: str
    gates: list[ReadinessGate]
    truth_boundary: str


def evaluate_training_readiness(
    *,
    audit: dict[str, Any],
    nccl_validation: dict[str, Any],
    operator_inputs: OperatorInputs,
    requirements: TrainingStackRequirements,
    readiness_version: str = READINESS_VERSION,
) -> TrainingReadinessResult:
    if not re.fullmatch(
        r"training-readiness-v[1-9]\d*", readiness_version
    ):
        raise ValueError(
            "readiness_version must match "
            "'training-readiness-v<positive integer>'"
        )
    results = {item["name"]: item for item in audit["results"]}
    gates = [
        _command_gate(
            results,
            name="gpu",
            gate_name="gpu_inventory",
            remediation="Restore nvidia-smi access on the target host.",
        ),
        _python_gate(results, requirements.python_minimum),
        _driver_gate(
            results,
            requirements,
            nccl_validation,
            operator_inputs.runtime_environment,
        ),
        _cuda_toolkit_gate(
            results,
            operator_inputs.cuda_toolkit,
            requirements.cuda_minimum,
        ),
        _confirmation_gate(
            "isolated_runtime",
            operator_inputs.runtime_environment.confirmed
            and bool(operator_inputs.runtime_environment.python_executable)
            and bool(operator_inputs.runtime_environment.torch_version)
            and bool(operator_inputs.runtime_environment.torch_cuda_version),
            (
                "Install and record an isolated PyTorch CUDA environment; "
                "do not use the system Python."
            ),
            operator_inputs.runtime_environment.reference,
        ),
        _confirmation_gate(
            "gpu_coordination",
            operator_inputs.gpu_reservation.confirmed,
            "Record the approved dynamic idle-GPU coordination rule.",
            operator_inputs.gpu_reservation.reference,
        ),
        _storage_gate(operator_inputs.storage),
        _student_gate(operator_inputs.student),
        _confirmation_gate(
            "git_sync",
            operator_inputs.git_sync.confirmed,
            "Record the reviewed local-to-server Git synchronization method.",
            operator_inputs.git_sync.reference,
        ),
        *_nccl_gates(nccl_validation, requirements.required_nccl_world_sizes),
    ]
    statuses = {gate.status for gate in gates}
    if GateStatus.BLOCKED in statuses:
        status = "blocked"
    elif GateStatus.PENDING in statuses:
        status = "pending"
    else:
        status = "ready"
    return TrainingReadinessResult(
        readiness_version=readiness_version,
        evaluated_at=datetime.now(UTC).isoformat(),
        status=status,
        gates=gates,
        truth_boundary=(
            "A ready result proves only that the declared Phase 0 gates have "
            "saved evidence. It does not prove model quality or authorize a "
            "training run on devices that fail the dynamic idle preflight."
        ),
    )


def write_readiness_artifacts(
    result: TrainingReadinessResult,
    *,
    requirements: TrainingStackRequirements,
    operator_inputs: OperatorInputs,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "readiness.json": (
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n"
        ),
        "requirements.json": (
            json.dumps(requirements.model_dump(mode="json"), indent=2) + "\n"
        ),
        "operator-inputs.json": (
            json.dumps(operator_inputs.model_dump(mode="json"), indent=2)
            + "\n"
        ),
        "report.md": render_readiness_report(result),
    }
    hashes: dict[str, str] = {}
    for name, content in payloads.items():
        encoded = content.encode("utf-8")
        (output_dir / name).write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "readiness_version": result.readiness_version,
        "status": result.status,
        "artifacts": hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def render_readiness_report(result: TrainingReadinessResult) -> str:
    lines = [
        "# Phase 0 training readiness",
        "",
        f"- Status: `{result.status}`",
        f"- Evaluated at: `{result.evaluated_at}`",
        "",
        "| Gate | Status | Evidence | Remediation |",
        "| --- | --- | --- | --- |",
    ]
    for gate in result.gates:
        lines.append(
            f"| `{gate.name}` | {gate.status.value} | "
            f"{_table_text(gate.evidence)} | "
            f"{_table_text(gate.remediation or '')} |"
        )
    lines.extend(["", result.truth_boundary, ""])
    return "\n".join(lines)


def _command_gate(
    results: dict[str, dict[str, Any]],
    *,
    name: str,
    gate_name: str,
    remediation: str,
) -> ReadinessGate:
    result = results.get(name)
    passed = result is not None and result["status"] == "succeeded"
    evidence = (
        f"audit check {name}={result['status']}"
        if result is not None
        else f"audit check {name}=missing"
    )
    return ReadinessGate(
        name=gate_name,
        status=GateStatus.PASSED if passed else GateStatus.BLOCKED,
        evidence=evidence,
        remediation=None if passed else remediation,
    )


def _python_gate(
    results: dict[str, dict[str, Any]],
    minimum: str,
) -> ReadinessGate:
    result = results.get("python")
    observed = _extract_version(result["stdout"]) if result else None
    passed = (
        result is not None
        and result["status"] == "succeeded"
        and observed is not None
        and _version_tuple(observed) >= _version_tuple(minimum)
    )
    return ReadinessGate(
        name="system_python",
        status=GateStatus.PASSED if passed else GateStatus.BLOCKED,
        evidence=f"observed={observed or 'unknown'}, required>={minimum}",
        remediation=None if passed else "Provide Python at the required version.",
    )


def _driver_gate(
    results: dict[str, dict[str, Any]],
    requirements: TrainingStackRequirements,
    nccl_validation: dict[str, Any],
    runtime: RuntimeConfirmation,
) -> ReadinessGate:
    gpu = results.get("gpu")
    observed = _extract_driver_version(gpu["stdout"]) if gpu else None
    if observed is None:
        return ReadinessGate(
            name="cuda_driver",
            status=GateStatus.BLOCKED,
            evidence="driver version unavailable",
            remediation="Restore nvidia-smi and collect the driver version.",
        )
    driver = _version_tuple(observed)
    full = _version_tuple(requirements.cuda_12_8_full_driver_minimum)
    minor = _version_tuple(
        requirements.cuda_12_minor_compatibility_driver_minimum
    )
    if driver >= full:
        return ReadinessGate(
            name="cuda_driver",
            status=GateStatus.PASSED,
            evidence=(
                f"driver={observed}, full CUDA {requirements.cuda_minimum} "
                f"driver>={requirements.cuda_12_8_full_driver_minimum}"
            ),
        )
    if driver >= minor:
        multi_gpu_nccl = next(
            (
                row
                for row in nccl_validation.get("planned_checks", [])
                if row.get("status") == "succeeded"
                and int(row.get("world_size", 0)) >= 2
            ),
            None,
        )
        if runtime.confirmed and runtime.reference and multi_gpu_nccl:
            return ReadinessGate(
                name="cuda_driver",
                status=GateStatus.PASSED,
                evidence=(
                    f"driver={observed} validated through exact-stack CUDA "
                    f"runtime evidence={runtime.reference} and NCCL "
                    f"world_size={multi_gpu_nccl['world_size']}"
                ),
            )
        return ReadinessGate(
            name="cuda_driver",
            status=GateStatus.PENDING,
            evidence=(
                f"driver={observed} is in the CUDA 12.x minor-compatibility "
                "range but below the full CUDA 12.8 driver"
            ),
            remediation=(
                "Either upgrade the driver or prove the exact pinned stack "
                "with CUDA tensor, extension, and NCCL runtime smokes."
            ),
        )
    return ReadinessGate(
        name="cuda_driver",
        status=GateStatus.BLOCKED,
        evidence=(
            f"driver={observed}, CUDA 12.x minor compatibility requires "
            f">={requirements.cuda_12_minor_compatibility_driver_minimum}"
        ),
        remediation="Upgrade the NVIDIA driver before preparing CUDA 12.x.",
    )


def _cuda_toolkit_gate(
    results: dict[str, dict[str, Any]],
    isolated: CudaToolkitConfirmation,
    minimum: str,
) -> ReadinessGate:
    result = results.get("cuda_toolkit")
    observed = None
    if result is not None:
        match = re.search(
            r"\brelease\s+(\d+(?:\.\d+){1,2})",
            result["stdout"],
            re.IGNORECASE,
        )
        observed = match.group(1) if match else None
    system_passed = (
        result is not None
        and result["status"] == "succeeded"
        and observed is not None
        and _version_tuple(observed) >= _version_tuple(minimum)
    )
    isolated_version = (
        _extract_version(isolated.version) if isolated.version else None
    )
    isolated_passed = (
        isolated.confirmed
        and isolated_version is not None
        and _version_tuple(isolated_version) >= _version_tuple(minimum)
        and bool(isolated.executable)
        and bool(isolated.compile_target)
        and bool(re.fullmatch(r"[0-9a-f]{64}", isolated.compiler_smoke_sha256))
        and bool(isolated.reference)
    )
    passed = system_passed or isolated_passed
    if system_passed:
        evidence = (
            f"system nvcc observed={observed}, required>={minimum}"
        )
    elif isolated_passed:
        evidence = (
            f"isolated nvcc observed={isolated_version}, "
            f"target={isolated.compile_target}, "
            f"smoke_sha256={isolated.compiler_smoke_sha256}, "
            f"reference={isolated.reference}, required>={minimum}"
        )
    else:
        isolated_observed = isolated_version or "unavailable"
        evidence = (
            f"system observed={observed or 'unavailable'}, "
            f"isolated observed={isolated_observed}, required>={minimum}"
        )
    return ReadinessGate(
        name="cuda_toolkit",
        status=GateStatus.PASSED if passed else GateStatus.BLOCKED,
        evidence=evidence,
        remediation=(
            None
            if passed
            else (
                "Prepare the pinned CUDA runtime/toolkit in an isolated "
                "environment and rerun the audit."
            )
        ),
    )


def _confirmation_gate(
    name: str,
    confirmed: bool,
    remediation: str,
    reference: str,
) -> ReadinessGate:
    return ReadinessGate(
        name=name,
        status=GateStatus.PASSED if confirmed else GateStatus.PENDING,
        evidence=reference if confirmed else "operator confirmation not recorded",
        remediation=None if confirmed else remediation,
    )


def _storage_gate(storage: StorageConfirmation) -> ReadinessGate:
    paths = [
        storage.project_root,
        storage.model_root,
        storage.data_root,
        storage.checkpoint_root,
    ]
    passed = storage.confirmed and all(paths)
    return ReadinessGate(
        name="durable_storage",
        status=GateStatus.PASSED if passed else GateStatus.PENDING,
        evidence=(
            storage.reference
            if passed
            else "four durable roots have not been confirmed"
        ),
        remediation=(
            None
            if passed
            else "Record writable project, model, data, and checkpoint roots."
        ),
    )


def _student_gate(student: StudentSelection) -> ReadinessGate:
    revision_is_immutable = bool(
        re.fullmatch(r"[0-9a-fA-F]{40}", student.revision)
    )
    passed = bool(
        student.model_id and revision_is_immutable and student.reference
    )
    return ReadinessGate(
        name="student_revision",
        status=GateStatus.PASSED if passed else GateStatus.PENDING,
        evidence=(
            (
                f"{student.model_id}@{student.revision}; "
                f"reference={student.reference}"
            )
            if passed
            else (
                "model ID, 40-character immutable revision, and evidence "
                "reference are not all recorded"
            )
        ),
        remediation=(
            None
            if passed
            else "Select the Student model and pin an immutable revision."
        ),
    )


def _nccl_gates(
    validation: dict[str, Any],
    required_world_sizes: list[int],
) -> list[ReadinessGate]:
    rows = {
        int(item["world_size"]): item
        for item in validation.get("planned_checks", [])
    }
    gates: list[ReadinessGate] = []
    for world_size in required_world_sizes:
        row = rows.get(world_size)
        passed = row is not None and row.get("status") == "succeeded"
        status = row.get("status", "missing") if row else "missing"
        gates.append(
            ReadinessGate(
                name=f"nccl_world_size_{world_size}",
                status=(
                    GateStatus.PASSED if passed else GateStatus.PENDING
                ),
                evidence=f"saved NCCL result={status}",
                remediation=(
                    None
                    if passed
                    else (
                        f"Run and save the world-size {world_size} NCCL smoke "
                        "inside the approved reservation."
                    )
                ),
            )
        )
    return gates


def _extract_version(value: str) -> str | None:
    match = re.search(r"\d+(?:\.\d+){1,2}", value)
    return match.group(0) if match else None


def _extract_driver_version(value: str) -> str | None:
    match = re.search(r",\s*(\d+\.\d+\.\d+)\s*$", value, re.MULTILINE)
    return match.group(1) if match else None


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
