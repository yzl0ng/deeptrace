from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CUDA_SOURCE = (
    'extern "C" __global__ void add_one(float* x) { x[0] += 1.0f; }\n'
)
KEY_PACKAGES = (
    "cuda-nvcc",
    "cuda-nvcc_linux-64",
    "cuda-nvcc-tools",
    "cuda-nvcc-impl",
    "gcc_linux-64",
    "gxx_linux-64",
)


def run_cuda_toolkit_smoke(
    *,
    toolkit_root: Path,
    architecture: str = "sm_89",
) -> dict[str, Any]:
    nvcc = toolkit_root / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc not found below toolkit root: {toolkit_root}")
    version_result = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"\bV(\d+(?:\.\d+){1,2})", version_result.stdout)
    if match is None:
        raise RuntimeError("nvcc output does not contain a release version")

    with tempfile.TemporaryDirectory(prefix="searchlab-cuda-smoke-") as tmp:
        ptx_path = Path(tmp) / "add_one.ptx"
        compile_result = subprocess.run(
            [
                str(nvcc),
                "-x",
                "cu",
                f"-arch={architecture}",
                "-ptx",
                "-",
                "-o",
                str(ptx_path),
            ],
            check=True,
            input=CUDA_SOURCE,
            capture_output=True,
            text=True,
        )
        ptx = ptx_path.read_bytes()
    if not ptx or f".target {architecture}".encode() not in ptx:
        raise RuntimeError("nvcc did not emit PTX for the requested target")
    if b".entry add_one(" not in ptx:
        raise RuntimeError("compiled PTX does not contain the smoke kernel")

    return {
        "evidence_version": "cuda-toolkit-bootstrap-v1",
        # The remote training runtime is Python 3.10 and has no datetime.UTC.
        "verified_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "status": "succeeded",
        "toolkit_root": _sanitize_path(toolkit_root),
        "nvcc_executable": _sanitize_path(nvcc),
        "nvcc_version": match.group(1),
        "nvcc_output": version_result.stdout.strip(),
        "compile_target": architecture,
        "source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "ptx_bytes": len(ptx),
        "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
        "compile_stdout": compile_result.stdout.strip(),
        "compile_stderr": compile_result.stderr.strip(),
        "packages": _key_packages(toolkit_root),
        "gpu_workload": "not_run",
        "truth_boundary": (
            "This proves that the isolated nvcc compiled one minimal CUDA "
            "kernel to PTX for the recorded architecture without running a "
            "GPU workload. It does not prove that every training extension "
            "will build or that multi-GPU training is ready."
        ),
    }


def write_cuda_toolkit_artifacts(
    evidence: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "evidence.json": json.dumps(evidence, indent=2) + "\n",
        "report.md": _render_report(evidence),
    }
    artifacts: list[dict[str, object]] = []
    for name, content in payloads.items():
        encoded = content.encode()
        (output_dir / name).write_bytes(encoded)
        artifacts.append(
            {
                "path": name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    manifest = {
        "evidence_version": evidence["evidence_version"],
        "status": evidence["status"],
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _key_packages(toolkit_root: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for metadata_path in (toolkit_root / "conda-meta").glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        name = metadata.get("name")
        if name in KEY_PACKAGES:
            packages[name] = (
                f"{metadata['version']}={metadata.get('build', 'unknown')}"
            )
    missing = sorted(set(KEY_PACKAGES) - set(packages))
    if missing:
        raise RuntimeError(f"toolchain metadata missing packages: {missing}")
    return dict(sorted(packages.items()))


def _sanitize_path(path: Path) -> str:
    value = str(path)
    username = os.environ.get("USER", "")
    if username:
        value = value.replace(f"/data/{username}/", "/data/$USER/")
        value = value.replace(f"/home/{username}/", "/home/$USER/")
    return value


def _render_report(evidence: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 0 isolated CUDA toolkit smoke",
            "",
            f"- Status: `{evidence['status']}`",
            f"- nvcc: `{evidence['nvcc_version']}`",
            f"- Toolkit: `{evidence['toolkit_root']}`",
            f"- Compile target: `{evidence['compile_target']}`",
            f"- PTX bytes: `{evidence['ptx_bytes']}`",
            f"- PTX SHA-256: `{evidence['ptx_sha256']}`",
            "- GPU workload: `not_run`",
            "",
            str(evidence["truth_boundary"]),
            "",
        ]
    )
