from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.agentic.runtime_smoke_plan import (
    build_runtime_smoke_command,
    run_runtime_smoke_plan,
    write_runtime_smoke_plan,
)


def test_runtime_smoke_command_keeps_gpu_selection_in_environment() -> None:
    command = build_runtime_smoke_command(
        mode="vllm_inference",
        python_executable="/runtime/python",
        model_path="/models/student",
        result_path=Path("vllm.json"),
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    assert command[:3] == [
        "/runtime/python",
        "-m",
        "app.agentic.gpu_runtime_smoke",
    ]
    assert "--model-path" in command
    assert "CUDA_VISIBLE_DEVICES" not in command


def test_runtime_smoke_execution_requires_auditable_reservation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="reservation-reference"):
        run_runtime_smoke_plan(
            python_executable="python",
            model_path="/models/student",
            output_dir=tmp_path,
            device=5,
            timeout_seconds=30,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            execute=True,
            reservation_confirmed=True,
            reservation_reference="",
        )


def test_runtime_smoke_plan_launches_nothing_and_is_hashable(
    tmp_path: Path,
) -> None:
    payload = run_runtime_smoke_plan(
        python_executable="python",
        model_path="/models/student",
        output_dir=tmp_path,
        device=5,
        timeout_seconds=30,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        execute=False,
        reservation_confirmed=False,
        reservation_reference="",
    )
    write_runtime_smoke_plan(payload, tmp_path)

    assert payload["status"] == "planned"
    assert [row["mode"] for row in payload["checks"]] == [
        "cuda_extensions",
        "vllm_inference",
    ]
    assert not (tmp_path / "cuda_extensions.json").exists()
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    expected = manifest["artifacts"]["runtime-validation.json"]
    assert expected == hashlib.sha256(
        (tmp_path / "runtime-validation.json").read_bytes()
    ).hexdigest()


def test_runtime_smoke_rejects_invalid_device_and_memory_fraction(
    tmp_path: Path,
) -> None:
    kwargs = {
        "python_executable": "python",
        "model_path": "/models/student",
        "output_dir": tmp_path,
        "device": -1,
        "timeout_seconds": 30,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "execute": False,
        "reservation_confirmed": False,
        "reservation_reference": "",
    }
    with pytest.raises(ValueError, match="device"):
        run_runtime_smoke_plan(**kwargs)

    kwargs["device"] = 5
    kwargs["gpu_memory_utilization"] = 1.0
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        run_runtime_smoke_plan(**kwargs)
