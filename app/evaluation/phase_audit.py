from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PhaseAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    status: str
    checks: dict[str, bool]
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class CompletedPhasesAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_version: str = "phase0-4-gap-audit-v1"
    generated_at: str
    overall_status: str
    phases: list[PhaseAuditEntry]
    truth_boundary: str


def audit_completed_phases(project_root: Path) -> CompletedPhasesAudit:
    phases = [
        _audit_phase_zero(project_root),
        _audit_phase_one(project_root),
        _audit_phase_two(project_root),
        _audit_phase_three(project_root),
        _audit_phase_four(project_root),
    ]
    overall_status = (
        "complete"
        if all(item.status == "complete" for item in phases)
        else "blocked"
        if any(item.status == "blocked" for item in phases)
        else "incomplete"
    )
    return CompletedPhasesAudit(
        generated_at=datetime.now(UTC).isoformat(),
        overall_status=overall_status,
        phases=phases,
        truth_boundary=(
            "This audit verifies saved artifacts and the original Phase 0-4 "
            "acceptance evidence. It does not upgrade scripted experiments "
            "into model-quality claims or bypass external GPU authorization."
        ),
    )


def write_phase_audit(
    audit: CompletedPhasesAudit,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "audit.json": (
            json.dumps(audit.model_dump(mode="json"), indent=2) + "\n"
        ),
        "report.md": _render_report(audit),
    }
    hashes: dict[str, str] = {}
    for name, content in payloads.items():
        encoded = content.encode("utf-8")
        (output_dir / name).write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "audit_version": audit.audit_version,
        "overall_status": audit.overall_status,
        "artifacts": hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _audit_phase_zero(root: Path) -> PhaseAuditEntry:
    readiness_path = (
        root / "data/system/training-readiness-v1/readiness.json"
    )
    required = [
        root / "upstream.lock.yaml",
        root / "THIRD_PARTY.md",
        root / "NOTICE",
        root / "data/system/server-audit-v2/manifest.json",
        readiness_path,
    ]
    files_present = all(path.is_file() for path in required)
    readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
    readiness_status = str(readiness.get("status", "missing"))
    gates = readiness.get("gates", [])
    gaps = [
        f"{item['name']}: {item['status']}"
        for item in gates
        if item.get("status") != "passed"
    ]
    checks = {
        "provenance_files_present": files_present,
        "server_audit_hashes_valid": _manifest_valid(
            root / "data/system/server-audit-v2"
        ),
        "training_readiness_ready": readiness_status == "ready",
    }
    return PhaseAuditEntry(
        phase="phase_0",
        status="complete" if all(checks.values()) else "blocked",
        checks=checks,
        evidence=[
            "Pinned upstream provenance and redacted server-audit v2 exist.",
            f"Saved training readiness status={readiness_status}.",
        ],
        gaps=gaps,
    )


def _audit_phase_one(root: Path) -> PhaseAuditEntry:
    directory = root / "data/experiments/deep-research-baseline-v1"
    metrics = _read_json(directory / "metrics.json")
    checks = {
        "artifact_hashes_valid": _manifest_valid(directory),
        "run_completed": metrics.get("status") == "completed",
        "search_tool_called": int(metrics.get("search_calls", 0)) > 0,
        "trace_saved": (directory / "trajectories.jsonl").is_file(),
    }
    return _entry("phase_1", checks, directory)


def _audit_phase_two(root: Path) -> PhaseAuditEntry:
    directory = root / "data/experiments/web-evidence-baseline-v1"
    manifest = _read_json(directory / "manifest.json")
    metrics = _read_json(directory / "metrics.json")
    checks = {
        "artifact_hashes_valid": _manifest_valid(directory),
        "experiment_completed": manifest.get("status") == "completed",
        "documents_and_evidence_saved": (
            int(metrics.get("documents", 0)) > 0
            and int(metrics.get("evidence_returned", 0)) > 0
        ),
        "failure_semantics_exercised": (
            int(metrics.get("failed_pages", 0)) > 0
        ),
        "cache_semantics_exercised": (
            int(metrics.get("second_cache_hits", 0)) > 0
            and int(metrics.get("second_page_reads", 0))
            < int(metrics.get("first_page_reads", 0))
        ),
    }
    return _entry("phase_2", checks, directory)


def _audit_phase_three(root: Path) -> PhaseAuditEntry:
    directory = root / "data/experiments/supervisor-recovery-v1"
    metrics = _read_json(directory / "metrics.json")
    checks = {
        "artifact_hashes_valid": _manifest_valid(directory),
        "run_completed": metrics.get("status") == "completed",
        "bounded_parallel_multitask": (
            int(metrics.get("subtasks", 0)) >= 2
            and int(metrics.get("max_parallel_observed", 0))
            <= int(metrics.get("configured_parallel_limit", 0))
        ),
        "query_rewrite_exercised": (
            int(metrics.get("query_rewrites", 0)) > 0
        ),
        "resume_did_not_repeat_initial_search": (
            int(metrics.get("repeated_completed_initial_queries", -1)) == 0
        ),
        "traceable_memory_saved": (
            int(metrics.get("memory_evidence_ids", 0)) > 0
            and int(metrics.get("memory_tool_call_ids", 0)) > 0
        ),
        "checkpoint_saved": int(metrics.get("checkpoint_version", 0)) > 0,
    }
    return _entry("phase_3", checks, directory)


def _audit_phase_four(root: Path) -> PhaseAuditEntry:
    directory = root / "data/experiments/agentic-eval-baseline-v1"
    manifest = _read_json(directory / "manifest.json")
    metrics = _read_json(directory / "metrics.json")
    dataset_dir = root / "data/evaluation/agentic-search-v1"
    dataset_manifest = _read_json(dataset_dir / "manifest.json")
    dataset_path = dataset_dir / "test.jsonl"
    modes = {"direct", "rag", "react", "deep_research"}
    case_total = sum(
        int(metrics.get(mode, {}).get("cases", 0)) for mode in modes
    )
    checks = {
        "artifact_hashes_valid": _manifest_valid(directory),
        "experiment_completed": manifest.get("status") == "completed",
        "four_modes_24_cases": set(metrics) == modes and case_total == 24,
        "frozen_dataset_hash_valid": (
            dataset_path.is_file()
            and _sha256(dataset_path) == dataset_manifest.get("test_sha256")
        ),
        "dataset_sources_pinned": all(
            isinstance(item.get("revision"), str)
            and len(item["revision"]) == 40
            and bool(item.get("license"))
            and bool(item.get("upstream"))
            for item in dataset_manifest.get("sources", [])
        )
        and len(dataset_manifest.get("sources", [])) == 3,
        "test_not_used_for_tuning": (
            _read_json(directory / "config.json").get(
                "test_used_for_tuning"
            )
            is False
        ),
        "citation_presence_separate_from_support": (
            metrics.get("rag", {})
            .get("metrics", {})
            .get("citation_presence_rate")
            == 1.0
            and metrics.get("rag", {})
            .get("metrics", {})
            .get("claim_support_rate")
            == 0.0
        ),
    }
    return _entry("phase_4", checks, directory)


def _entry(
    phase: str,
    checks: dict[str, bool],
    evidence_dir: Path,
) -> PhaseAuditEntry:
    gaps = [name for name, passed in checks.items() if not passed]
    return PhaseAuditEntry(
        phase=phase,
        status="complete" if not gaps else "incomplete",
        checks=checks,
        evidence=[f"data/experiments/{evidence_dir.name}"],
        gaps=gaps,
    )


def _manifest_valid(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        expected = {
            str(item["path"]): str(item["sha256"])
            for item in artifacts
            if isinstance(item, dict)
            and "path" in item
            and "sha256" in item
        }
    elif isinstance(artifacts, dict):
        expected = {
            str(name): str(digest) for name, digest in artifacts.items()
        }
    else:
        return False
    return bool(expected) and all(
        (directory / name).is_file()
        and _sha256(directory / name) == digest
        for name, digest in expected.items()
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_report(audit: CompletedPhasesAudit) -> str:
    lines = [
        "# Phase 0-4 gap audit",
        "",
        f"- Overall status: `{audit.overall_status}`",
        f"- Generated at: `{audit.generated_at}`",
        "",
        "| Phase | Status | Passed checks | Total checks |",
        "| --- | --- | ---: | ---: |",
    ]
    for phase in audit.phases:
        passed = sum(phase.checks.values())
        lines.append(
            f"| `{phase.phase}` | {phase.status} | "
            f"{passed} | {len(phase.checks)} |"
        )
    lines.extend(["", "## Remaining gaps", ""])
    for phase in audit.phases:
        if phase.gaps:
            lines.append(f"- `{phase.phase}`: " + "; ".join(phase.gaps))
    lines.extend(["", audit.truth_boundary, ""])
    return "\n".join(lines)
