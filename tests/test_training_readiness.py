from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.agentic.nccl_matrix import (
    NCCL_MATRIX,
    build_torchrun_command,
    discover_idle_devices,
    run_nccl_matrix,
    write_nccl_matrix,
)
from app.agentic.training_readiness import (
    Confirmation,
    CudaToolkitConfirmation,
    GateStatus,
    OperatorInputs,
    RuntimeConfirmation,
    StorageConfirmation,
    StudentSelection,
    TrainingStackRequirements,
    evaluate_training_readiness,
    write_readiness_artifacts,
)


def _requirements() -> TrainingStackRequirements:
    return TrainingStackRequirements(
        schema_version=1,
        agent_r1_commit="a" * 40,
        verl_commit="b" * 40,
        verl_tag="v0.7.0",
        python_minimum="3.10",
        cuda_minimum="12.8",
        cuda_12_minor_compatibility_driver_minimum="525.60.13",
        cuda_12_8_full_driver_minimum="570.26",
        required_nccl_world_sizes=[1, 4, 8],
    )


def _audit(
    *,
    driver: str = "535.309.01",
    toolkit_status: str = "failed",
) -> dict[str, object]:
    return {
        "results": [
            {
                "name": "gpu",
                "status": "succeeded",
                "stdout": (
                    "0, NVIDIA GeForce RTX 4090, 24564 MiB, "
                    f"11 MiB, {driver}"
                ),
            },
            {
                "name": "python",
                "status": "succeeded",
                "stdout": "Python 3.10.12",
            },
            {
                "name": "cuda_toolkit",
                "status": toolkit_status,
                "stdout": (
                    "Cuda compilation tools, release 12.8"
                    if toolkit_status == "succeeded"
                    else ""
                ),
            },
        ]
    }


def _nccl(status: str) -> dict[str, object]:
    return {
        "planned_checks": [
            {"world_size": world_size, "status": status}
            for world_size in (1, 4, 8)
        ]
    }


def _confirmed_inputs() -> OperatorInputs:
    return OperatorInputs(
        gpu_reservation=Confirmation(
            confirmed=True,
            reference="approved-window",
        ),
        runtime_environment=RuntimeConfirmation(
            confirmed=True,
            reference="runtime-manifest",
            python_executable="/project/.venv/bin/python",
            torch_version="2.8.0",
            torch_cuda_version="12.8",
        ),
        cuda_toolkit=CudaToolkitConfirmation(
            confirmed=True,
            reference="cuda-toolkit-smoke",
            version="12.8.93",
            executable="/toolchains/cuda-12.8/bin/nvcc",
            compile_target="sm_89",
            compiler_smoke_sha256="c" * 64,
        ),
        storage=StorageConfirmation(
            confirmed=True,
            reference="storage-check",
            project_root="/project",
            model_root="/models",
            data_root="/datasets",
            checkpoint_root="/checkpoints",
        ),
        student=StudentSelection(
            model_id="org/student",
            revision="f" * 40,
            reference="student-lock",
        ),
        git_sync=Confirmation(
            confirmed=True,
            reference="reviewed-sync",
        ),
    )


def test_current_driver_is_candidate_not_claimed_fully_compatible() -> None:
    result = evaluate_training_readiness(
        audit=_audit(),
        nccl_validation=_nccl("not_run"),
        operator_inputs=OperatorInputs(),
        requirements=_requirements(),
    )
    gates = {gate.name: gate for gate in result.gates}

    assert result.status == "blocked"
    assert gates["cuda_driver"].status == GateStatus.PENDING
    assert "minor-compatibility" in gates["cuda_driver"].evidence
    assert gates["cuda_toolkit"].status == GateStatus.BLOCKED
    assert gates["gpu_coordination"].status == GateStatus.PENDING


def test_minor_compatibility_driver_passes_with_exact_runtime_and_world_two() -> None:
    nccl = {
        "planned_checks": [
            {"world_size": 1, "status": "succeeded"},
            {"world_size": 2, "status": "succeeded"},
        ]
    }
    result = evaluate_training_readiness(
        audit=_audit(),
        nccl_validation=nccl,
        operator_inputs=_confirmed_inputs(),
        requirements=_requirements(),
    )
    gates = {gate.name: gate for gate in result.gates}

    assert gates["cuda_driver"].status == GateStatus.PASSED
    assert "NCCL world_size=2" in gates["cuda_driver"].evidence


def test_all_saved_evidence_can_reach_ready() -> None:
    result = evaluate_training_readiness(
        audit=_audit(driver="570.26.00", toolkit_status="succeeded"),
        nccl_validation=_nccl("succeeded"),
        operator_inputs=_confirmed_inputs(),
        requirements=_requirements(),
    )

    assert result.status == "ready"
    assert all(gate.status == GateStatus.PASSED for gate in result.gates)


def test_student_gate_requires_immutable_revision_and_reference() -> None:
    inputs = _confirmed_inputs()
    inputs.student.revision = "main"

    result = evaluate_training_readiness(
        audit=_audit(driver="570.26.00", toolkit_status="succeeded"),
        nccl_validation=_nccl("succeeded"),
        operator_inputs=inputs,
        requirements=_requirements(),
    )
    gates = {gate.name: gate for gate in result.gates}

    assert result.status == "pending"
    assert gates["student_revision"].status == GateStatus.PENDING


def test_readiness_version_is_explicit_and_validated() -> None:
    result = evaluate_training_readiness(
        audit=_audit(),
        nccl_validation=_nccl("not_run"),
        operator_inputs=OperatorInputs(),
        requirements=_requirements(),
        readiness_version="training-readiness-v2",
    )
    assert result.readiness_version == "training-readiness-v2"

    with pytest.raises(ValueError, match="readiness_version"):
        evaluate_training_readiness(
            audit=_audit(),
            nccl_validation=_nccl("not_run"),
            operator_inputs=OperatorInputs(),
            requirements=_requirements(),
            readiness_version="../../invalid",
        )


def test_outdated_cuda_toolkit_does_not_pass() -> None:
    audit = _audit(driver="570.26.00", toolkit_status="succeeded")
    audit["results"][2]["stdout"] = (
        "Cuda compilation tools, release 12.2, V12.2.140"
    )
    inputs = _confirmed_inputs()
    inputs.cuda_toolkit = CudaToolkitConfirmation()

    result = evaluate_training_readiness(
        audit=audit,
        nccl_validation=_nccl("succeeded"),
        operator_inputs=inputs,
        requirements=_requirements(),
    )
    gates = {gate.name: gate for gate in result.gates}

    assert result.status == "blocked"
    assert gates["cuda_toolkit"].status == GateStatus.BLOCKED
    assert "observed=12.2" in gates["cuda_toolkit"].evidence


def test_isolated_cuda_toolkit_can_replace_missing_system_nvcc() -> None:
    result = evaluate_training_readiness(
        audit=_audit(driver="570.26.00"),
        nccl_validation=_nccl("succeeded"),
        operator_inputs=_confirmed_inputs(),
        requirements=_requirements(),
    )
    gates = {gate.name: gate for gate in result.gates}

    assert result.status == "ready"
    assert gates["cuda_toolkit"].status == GateStatus.PASSED
    assert "isolated nvcc observed=12.8.93" in gates["cuda_toolkit"].evidence


def test_readiness_writer_hashes_every_artifact(tmp_path: Path) -> None:
    requirements = _requirements()
    inputs = OperatorInputs()
    result = evaluate_training_readiness(
        audit=_audit(),
        nccl_validation=_nccl("not_run"),
        operator_inputs=inputs,
        requirements=requirements,
    )

    write_readiness_artifacts(
        result,
        requirements=requirements,
        operator_inputs=inputs,
        output_dir=tmp_path,
    )

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected


def test_nccl_matrix_builds_expected_torchrun_command() -> None:
    command = build_torchrun_command(
        NCCL_MATRIX[1],
        python_executable="/runtime/python",
        result_path=Path("four.json"),
        timeout_seconds=90,
    )

    assert command[:3] == [
        "/runtime/python",
        "-m",
        "torch.distributed.run",
    ]
    assert "--nproc-per-node=4" in command
    assert "app.agentic.distributed_smoke" in command


def test_nccl_discovery_excludes_busy_and_high_memory_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            "0, GPU-a, 11\n1, GPU-b, 11\n2, GPU-c, 128\n",
            "GPU-b\n",
        ]
    )

    def fake_run(*args: object, **kwargs: object) -> object:
        class Result:
            stdout = next(outputs)

        return Result()

    monkeypatch.setattr("app.agentic.nccl_matrix.subprocess.run", fake_run)

    assert discover_idle_devices(max_used_memory_mib=64) == [0]


def test_nccl_discovery_can_allow_low_memory_compute_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            "0, GPU-a, 410\n1, GPU-b, 21417\n2, GPU-c, 719\n",
            "GPU-a\nGPU-c\n",
        ]
    )

    def fake_run(*args: object, **kwargs: object) -> object:
        class Result:
            stdout = next(outputs)

        return Result()

    monkeypatch.setattr("app.agentic.nccl_matrix.subprocess.run", fake_run)

    assert discover_idle_devices(
        max_used_memory_mib=2048,
        require_no_compute_processes=False,
    ) == [0, 2]


def test_nccl_execution_waits_when_too_few_gpus_are_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.agentic.nccl_matrix.discover_idle_devices",
        lambda **_: [5, 7],
    )
    payload = run_nccl_matrix(
        selected_names=["four_gpu"],
        python_executable="python",
        output_dir=tmp_path,
        timeout_seconds=30,
        execute=True,
    )

    assert payload["status"] == "pending"
    assert payload["planned_checks"][0]["status"] == "pending"
    assert payload["planned_checks"][0]["idle_devices_at_preflight"] == [5, 7]


def test_nccl_plan_launches_nothing_and_is_hashable(tmp_path: Path) -> None:
    payload = run_nccl_matrix(
        selected_names=[check.name for check in NCCL_MATRIX],
        python_executable="python",
        output_dir=tmp_path,
        timeout_seconds=120,
        execute=False,
    )
    write_nccl_matrix(payload, tmp_path)

    assert payload["status"] == "planned"
    assert [row["world_size"] for row in payload["planned_checks"]] == [
        1,
        4,
        8,
    ]
    assert not (tmp_path / "single_gpu.json").exists()
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    expected = manifest["artifacts"]["nccl-validation.json"]
    assert expected == hashlib.sha256(
        (tmp_path / "nccl-validation.json").read_bytes()
    ).hexdigest()


def test_nccl_writer_hashes_executed_rank_zero_result(
    tmp_path: Path,
) -> None:
    payload = run_nccl_matrix(
        selected_names=["single_gpu"],
        python_executable="python3",
        output_dir=tmp_path,
        timeout_seconds=120,
        execute=False,
    )
    payload["planned_checks"][0]["status"] = "succeeded"
    (tmp_path / "single_gpu.json").write_text(
        '{"status":"succeeded"}\n',
        encoding="utf-8",
    )

    write_nccl_matrix(payload, tmp_path)

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert "single_gpu.json" in manifest["artifacts"]
