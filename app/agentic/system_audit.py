from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agentic.sanitization import sanitize_text

AUDIT_VERSION = "server-audit-v4"

DEFAULT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("os_release", "cat /etc/os-release"),
    ("kernel", "uname -srmo"),
    ("cpu", "lscpu"),
    ("memory", "free -h"),
    ("disk", "df -h / /data"),
    (
        "gpu",
        (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
            "driver_version --format=csv,noheader"
        ),
    ),
    ("nvidia_smi_header", "nvidia-smi | sed -n '1,3p'"),
    ("gpu_topology", "nvidia-smi topo -m"),
    ("cuda_toolkit", "nvcc --version"),
    ("python", "python3 --version"),
    ("python_packages", "python3 -m pip show torch transformers"),
    (
        "environment_manager",
        "command -v conda || command -v micromamba || command -v uv",
    ),
    ("git", "git --version"),
    ("docker", "docker --version"),
    (
        "docker_runtime",
        (
            "docker info --format '{{json .Runtimes}}' 2>/dev/null "
            "| grep -q '\"nvidia\"'"
        ),
    ),
    (
        "writable_roots",
        (
            'for path in "$HOME" /data; do '
            'if test -d "$path" && test -w "$path"; then '
            'printf "%s=writable\\n" "$path"; '
            'else printf "%s=not_writable\\n" "$path"; fi; done'
        ),
    ),
    ("nccl_tests", "command -v all_reduce_perf"),
    (
        "scheduler",
        (
            "found=0; "
            "for name in squeue sbatch srun sinfo qstat qsub bsub; do "
            "if command -v \"$name\" >/dev/null 2>&1; then "
            "printf '%s=present\\n' \"$name\"; found=1; "
            "fi; done; "
            "if test \"$found\" -eq 0; then printf 'none\\n'; exit 1; fi"
        ),
    ),
    (
        "python_venv",
        "python3 -m venv --help >/dev/null && python3 -m pip --version",
    ),
    (
        "build_toolchain",
        (
            "for name in gcc g++ make cmake; do "
            "command -v \"$name\" >/dev/null 2>&1 || exit 1; "
            "done; "
            "printf 'gcc,g++,make,cmake=present\\n'; "
            "if command -v ninja >/dev/null 2>&1; then "
            "printf 'ninja=present\\n'; else printf 'ninja=absent\\n'; fi"
        ),
    ),
    (
        "storage_quota",
        (
            "quota -s 2>/dev/null | "
            "sed -E 's/^Disk quotas for user .*$/"
            "Disk quotas for current user/'"
        ),
    ),
    (
        "durable_root_candidates",
        (
            "for item in "
            "\"home:$HOME\" "
            "\"data_user:/data/$USER\" "
            "\"data_users:/data/users/$USER\" "
            "\"scratch_user:/scratch/$USER\" "
            "\"workspace_user:/workspace/$USER\"; do "
            "label=${item%%:*}; path=${item#*:}; "
            "if test -d \"$path\"; then "
            "writable=no; test -w \"$path\" && writable=yes; "
            "stats=$(df -k --output=avail,pcent,target \"$path\" "
            "| tail -1 | xargs); "
            "printf '%s exists=yes writable=%s %s\\n' "
            "\"$label\" \"$writable\" \"$stats\"; "
            "else printf '%s exists=no writable=no\\n' \"$label\"; fi; "
            "done"
        ),
    ),
    (
        "student_cache",
        (
            "root=\"$HOME/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-8B\"; "
            "if test -f \"$root/refs/main\"; then "
            "printf 'main_revision='; cat \"$root/refs/main\"; "
            "else printf 'main_revision=absent\\n'; exit 1; fi"
        ),
    ),
)

@dataclass(frozen=True)
class CommandResult:
    name: str
    command: str
    status: str
    return_code: int | None
    duration_ms: float
    stdout: str
    stderr: str


Runner = Callable[[str, str, str | None, float], CommandResult]


def run_command(
    name: str,
    command: str,
    ssh_host: str | None,
    timeout_seconds: float,
) -> CommandResult:
    argv = ["ssh", ssh_host, command] if ssh_host else ["bash", "-lc", command]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        return CommandResult(
            name=name,
            command=command,
            status="timed_out",
            return_code=None,
            duration_ms=round(duration_ms, 3),
            stdout=sanitize_text(exc.stdout or ""),
            stderr=sanitize_text(exc.stderr or ""),
        )

    duration_ms = (time.perf_counter() - started) * 1000
    return CommandResult(
        name=name,
        command=command,
        status="succeeded" if completed.returncode == 0 else "failed",
        return_code=completed.returncode,
        duration_ms=round(duration_ms, 3),
        stdout=sanitize_text(completed.stdout.strip()),
        stderr=sanitize_text(completed.stderr.strip()),
    )


def collect_audit(
    *,
    ssh_host: str | None,
    commands: Iterable[tuple[str, str]] = DEFAULT_COMMANDS,
    timeout_seconds: float = 20,
    runner: Runner = run_command,
    audit_version: str = AUDIT_VERSION,
) -> dict[str, Any]:
    if not re.fullmatch(r"server-audit-v[1-9]\d*", audit_version):
        raise ValueError(
            "audit_version must match 'server-audit-v<positive integer>'"
        )
    results = [
        runner(name, command, ssh_host, timeout_seconds)
        for name, command in commands
    ]
    return {
        "audit_version": audit_version,
        "collected_at": datetime.now(UTC).isoformat(),
        "target": "remote_ssh_alias" if ssh_host else "local",
        "results": [asdict(result) for result in results],
    }


def render_report(audit: dict[str, Any]) -> str:
    result_lines = [
        (
            "# Server audit "
            f"{audit['audit_version'].removeprefix('server-audit-')}"
        ),
        "",
        f"- Collected at: `{audit['collected_at']}`",
        f"- Target: `{audit['target']}`",
        "- Secrets, home-directory usernames, and explicit SSH targets are redacted.",
        "",
        "| Check | Status | Exit code | Duration (ms) |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in audit["results"]:
        exit_code = (
            "n/a" if result["return_code"] is None else str(result["return_code"])
        )
        result_lines.append(
            f"| `{result['name']}` | {result['status']} | {exit_code} | "
            f"{result['duration_ms']:.3f} |"
        )
    result_lines.extend(
        [
            "",
            (
                "Detailed command output is stored in `environment.json`. "
                "A failed optional check is evidence of an unavailable "
                "capability, not a successful validation."
            ),
            "",
        ]
    )
    return "\n".join(result_lines)


def write_audit(audit: dict[str, Any], output_dir: Path) -> None:
    import hashlib
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "environment.json": json.dumps(audit, ensure_ascii=False, indent=2)
        + "\n",
        "report.md": render_report(audit),
    }
    manifest_artifacts: list[dict[str, Any]] = []
    for filename, content in artifacts.items():
        encoded = content.encode("utf-8")
        (output_dir / filename).write_bytes(encoded)
        manifest_artifacts.append(
            {
                "path": filename,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    for artifact_path in sorted(output_dir.iterdir()):
        if (
            not artifact_path.is_file()
            or artifact_path.name == "manifest.json"
            or artifact_path.name in artifacts
        ):
            continue
        encoded = artifact_path.read_bytes()
        manifest_artifacts.append(
            {
                "path": artifact_path.name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    manifest = {
        "audit_version": audit["audit_version"],
        "collected_at": audit["collected_at"],
        "artifacts": manifest_artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
