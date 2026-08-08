from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.agentic.cuda_toolkit_smoke import write_cuda_toolkit_artifacts


def test_cuda_toolkit_writer_hashes_artifacts(tmp_path: Path) -> None:
    evidence = {
        "evidence_version": "cuda-toolkit-bootstrap-v1",
        "verified_at": "2026-07-29T00:00:00+00:00",
        "status": "succeeded",
        "toolkit_root": "/data/$USER/toolkit",
        "nvcc_executable": "/data/$USER/toolkit/bin/nvcc",
        "nvcc_version": "12.8.93",
        "compile_target": "sm_89",
        "ptx_bytes": 491,
        "ptx_sha256": "a" * 64,
        "gpu_workload": "not_run",
        "truth_boundary": "compiler-only evidence",
    }

    write_cuda_toolkit_artifacts(evidence, tmp_path)

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    for item in manifest["artifacts"]:
        artifact = tmp_path / item["path"]
        assert artifact.stat().st_size == item["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == item["sha256"]
