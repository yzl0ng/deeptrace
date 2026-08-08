from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.phase_audit import (
    _manifest_valid,
    audit_completed_phases,
    write_phase_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_EXPERIMENTS_AVAILABLE = (
    PROJECT_ROOT / "data/experiments/deep-research-baseline-v1/metrics.json"
).exists()


@pytest.mark.skipif(
    not SAVED_EXPERIMENTS_AVAILABLE,
    reason="archived experiment artifacts are excluded from the public repo",
)
def test_saved_phase_one_through_four_acceptance_evidence_is_complete() -> None:
    audit = audit_completed_phases(PROJECT_ROOT)
    phases = {item.phase: item for item in audit.phases}

    assert phases["phase_0"].status == "blocked"
    assert phases["phase_0"].checks["training_readiness_ready"] is False
    for name in ("phase_1", "phase_2", "phase_3", "phase_4"):
        assert phases[name].status == "complete"
        assert all(phases[name].checks.values())
    assert audit.overall_status == "blocked"


@pytest.mark.skipif(
    not SAVED_EXPERIMENTS_AVAILABLE,
    reason="archived experiment artifacts are excluded from the public repo",
)
def test_phase_audit_writer_hashes_outputs_without_local_paths(
    tmp_path: Path,
) -> None:
    audit = audit_completed_phases(PROJECT_ROOT)

    write_phase_audit(audit, tmp_path)

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected
    serialized = (tmp_path / "audit.json").read_text(encoding="utf-8")
    assert str(PROJECT_ROOT) not in serialized


def test_manifest_verifier_detects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text('{"status":"completed"}\n', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifacts": {"result.json": digest}}),
        encoding="utf-8",
    )
    assert _manifest_valid(tmp_path) is True

    artifact.write_text('{"status":"changed"}\n', encoding="utf-8")
    assert _manifest_valid(tmp_path) is False
