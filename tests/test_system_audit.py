from __future__ import annotations

from pathlib import Path

from app.agentic.distributed_smoke import expected_all_reduce_sum
from app.agentic.system_audit import (
    DEFAULT_COMMANDS,
    CommandResult,
    collect_audit,
    render_report,
    sanitize_text,
    write_audit,
)


def test_default_audit_commands_cover_phase0_infrastructure() -> None:
    names = [name for name, _ in DEFAULT_COMMANDS]

    assert len(names) == len(set(names))
    assert {
        "scheduler",
        "python_venv",
        "build_toolchain",
        "storage_quota",
        "durable_root_candidates",
        "student_cache",
    } <= set(names)


def test_sanitize_text_removes_secrets_and_home_username() -> None:
    raw = (
        "DEEPSEEK_API_KEY=sk-sensitive\n"
        "cache=/home/private-user/.cache/model\n"
        "model=/data/private-user/models/student\n"
        "ssh deploy@example.com\n"
        "Disk quotas for user private-user (uid 1013):"
    )

    sanitized = sanitize_text(raw)

    assert "sk-sensitive" not in sanitized
    assert "private-user" not in sanitized
    assert "example.com" not in sanitized
    assert "uid 1013" not in sanitized
    assert "<redacted>" in sanitized
    assert "$HOME/.cache/model" in sanitized
    assert "/data/$USER/models/student" in sanitized
    assert "Disk quotas for current user" in sanitized


def test_collect_audit_records_failures_without_claiming_success() -> None:
    def fake_runner(
        name: str,
        command: str,
        ssh_host: str | None,
        timeout_seconds: float,
    ) -> CommandResult:
        assert ssh_host == "alias"
        assert timeout_seconds == 3
        return CommandResult(
            name=name,
            command=command,
            status="failed",
            return_code=127,
            duration_ms=1.5,
            stdout="",
            stderr="not found",
        )

    audit = collect_audit(
        ssh_host="alias",
        commands=(("cuda", "nvcc --version"),),
        timeout_seconds=3,
        runner=fake_runner,
        audit_version="server-audit-v3",
    )

    assert audit["audit_version"] == "server-audit-v3"
    assert audit["target"] == "remote_ssh_alias"
    assert audit["results"][0]["status"] == "failed"
    assert "failed" in render_report(audit)


def test_collect_audit_rejects_invalid_version() -> None:
    import pytest

    with pytest.raises(ValueError, match="audit_version"):
        collect_audit(
            ssh_host=None,
            commands=(),
            audit_version="../server-audit-v3",
        )


def test_write_audit_creates_json_and_report(tmp_path: Path) -> None:
    audit = {
        "audit_version": "server-audit-v1",
        "collected_at": "2026-07-29T00:00:00+00:00",
        "target": "local",
        "results": [],
    }

    write_audit(audit, tmp_path)

    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_write_audit_includes_existing_phase_artifacts_in_manifest(
    tmp_path: Path,
) -> None:
    import json

    (tmp_path / "nccl-validation.json").write_text("{}\n", encoding="utf-8")
    audit = {
        "audit_version": "server-audit-v1",
        "collected_at": "2026-07-29T00:00:00+00:00",
        "target": "local",
        "results": [],
    }

    write_audit(audit, tmp_path)

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert {
        artifact["path"] for artifact in manifest["artifacts"]
    } == {"environment.json", "report.md", "nccl-validation.json"}


def test_expected_all_reduce_sum() -> None:
    assert expected_all_reduce_sum(1) == 1.0
    assert expected_all_reduce_sum(4) == 10.0
    assert expected_all_reduce_sum(8) == 36.0
