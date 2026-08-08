from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def build_acceptance_checks(
    *,
    status: dict[str, Any],
    created: dict[str, Any],
    persisted: dict[str, Any],
    web_html: str | None,
) -> list[dict[str, Any]]:
    subtasks = created.get("subtasks") or []
    tool_calls = created.get("tool_calls") or []
    assessments = created.get("evidence_assessments") or []
    final_evidence_ids = {
        str(item) for item in created.get("final_evidence_ids") or []
    }
    latest_assessments: dict[str, dict[str, Any]] = {}
    for assessment in assessments:
        subtask_id = str(assessment.get("subtask_id", ""))
        previous = latest_assessments.get(subtask_id)
        if previous is None or int(assessment.get("attempt", 0)) >= int(
            previous.get("attempt", 0)
        ):
            latest_assessments[subtask_id] = assessment
    allowed_evidence_ids = {
        str(evidence_id)
        for assessment in latest_assessments.values()
        for evidence_id in assessment.get("evidence_ids") or []
    }
    checkpoint = created.get("checkpoint") or {}
    usage = created.get("usage") or {}

    checks = [
        _check(
            "api_ready",
            status.get("ready") is True,
            f"ready={status.get('ready')}",
        ),
        _check(
            "supervisor_workflow",
            status.get("workflow") == "supervisor",
            f"workflow={status.get('workflow')}",
        ),
        _check(
            "run_completed",
            created.get("status") == "completed"
            and created.get("stop_reason") == "completed",
            (
                f"status={created.get('status')}, "
                f"stop_reason={created.get('stop_reason')}"
            ),
        ),
        _check(
            "plan_created",
            len(created.get("plan") or []) >= 2,
            f"plan_items={len(created.get('plan') or [])}",
        ),
        _check(
            "subtasks_completed",
            bool(subtasks)
            and all(item.get("status") == "completed" for item in subtasks),
            ", ".join(
                f"{item.get('subtask_id')}={item.get('status')}"
                for item in subtasks
            ),
        ),
        _check(
            "tools_succeeded",
            bool(tool_calls)
            and all(item.get("status") == "succeeded" for item in tool_calls),
            f"tool_calls={len(tool_calls)}",
        ),
        _check(
            "evidence_sufficient",
            len(latest_assessments) == len(subtasks)
            and bool(latest_assessments)
            and all(
                item.get("sufficient") is True
                for item in latest_assessments.values()
            ),
            (
                f"latest_sufficient="
                f"{sum(item.get('sufficient') is True for item in latest_assessments.values())}"
                f"/{len(latest_assessments)}"
            ),
        ),
        _check(
            "evidence_allowlist",
            bool(final_evidence_ids)
            and final_evidence_ids.issubset(allowed_evidence_ids),
            (
                f"cited={len(final_evidence_ids)}, "
                f"allowed={len(allowed_evidence_ids)}"
            ),
        ),
        _check(
            "final_report_present",
            len(str(created.get("final_report") or "").strip()) >= 80,
            f"characters={len(str(created.get('final_report') or '').strip())}",
        ),
        _check(
            "checkpoint_completed",
            checkpoint.get("stage") == "report"
            and "report" in (checkpoint.get("completed_stages") or []),
            (
                f"stage={checkpoint.get('stage')}, "
                f"version={checkpoint.get('version')}"
            ),
        ),
        _check(
            "usage_recorded",
            int(usage.get("agent_steps", 0)) > 0
            and int(usage.get("search_calls", 0)) == len(tool_calls)
            and int(usage.get("total_tokens", 0)) > 0,
            (
                f"steps={usage.get('agent_steps')}, "
                f"search={usage.get('search_calls')}, "
                f"tokens={usage.get('total_tokens')}"
            ),
        ),
        _check(
            "no_runtime_errors",
            not (created.get("errors") or []),
            f"errors={len(created.get('errors') or [])}",
        ),
        _check(
            "persistence_round_trip",
            persisted.get("run_id") == created.get("run_id")
            and persisted.get("final_report") == created.get("final_report")
            and persisted.get("checkpoint") == created.get("checkpoint"),
            f"run_id={persisted.get('run_id')}",
        ),
    ]
    if web_html is not None:
        checks.append(
            _check(
                "web_control_surface",
                "Agent 全链路" in web_html
                and "SearchLab" in web_html,
                "SSR contains Agent navigation and SearchLab shell",
            )
        )
    return checks


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _write_report(
    output_dir: Path,
    *,
    query: str,
    status: dict[str, Any],
    run: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "v2-end-to-end-acceptance-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "query": query,
        "run_id": run.get("run_id"),
        "passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "runtime": {
            "workflow": status.get("workflow"),
            "search_provider": status.get("search_provider"),
            "model": run.get("model_name"),
            "usage": run.get("usage"),
        },
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(
        (
            f"| `{item['name']}` | "
            f"{'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['evidence']} |"
        )
        for item in checks
    )
    report = f"""# SearchLab v2 端到端验收

- Run ID: `{run.get("run_id")}`
- Query: {query}
- Result: `{"PASS" if summary["passed"] else "FAIL"}`
- Checks: `{summary["checks_passed"]}/{summary["checks_total"]}`
- Workflow: `{status.get("workflow")}`
- Search provider: `{status.get("search_provider")}`
- Model: `{run.get("model_name")}`

| Check | Result | Evidence |
|---|---:|---|
{rows}

## Final report

{run.get("final_report") or "No report generated."}

## Final evidence IDs

{json.dumps(run.get("final_evidence_ids") or [], ensure_ascii=False)}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and save the real SearchLab v2 end-to-end acceptance."
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--web-url")
    parser.add_argument(
        "--query",
        default=(
            "BM25 和 Dense Retrieval 为什么互补？"
            "RRF 为什么不应该直接相加两路原始分数？"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/v2-end-to-end-acceptance-v1"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    with httpx.Client(
        timeout=args.timeout_seconds,
        trust_env=False,
    ) as client:
        status_response = client.get(f"{api_base}/api/v2/research/status")
        status_response.raise_for_status()
        status = status_response.json()
        create_response = client.post(
            f"{api_base}/api/v2/research/runs",
            json={
                "query": args.query,
                "budget": {
                    "max_wall_time_seconds": 180,
                    "max_agent_steps": 24,
                    "max_search_calls": 8,
                    "max_page_reads": 0,
                    "max_total_tokens": 30_000,
                    "max_parallel_research_units": 3,
                },
            },
        )
        create_response.raise_for_status()
        created = create_response.json()
        persisted_response = client.get(
            f"{api_base}/api/v2/research/runs/{created['run_id']}"
        )
        persisted_response.raise_for_status()
        persisted = persisted_response.json()
        web_html = None
        if args.web_url:
            web_response = client.get(args.web_url)
            web_response.raise_for_status()
            web_html = web_response.text

    checks = build_acceptance_checks(
        status=status,
        created=created,
        persisted=persisted,
        web_html=web_html,
    )
    _write_report(
        args.output_dir,
        query=args.query,
        status=status,
        run=created,
        checks=checks,
    )
    passed = sum(item["passed"] for item in checks)
    print(
        f"v2 end-to-end acceptance: {passed}/{len(checks)} "
        f"to {args.output_dir}"
    )
    for item in checks:
        print(
            f"[{'PASS' if item['passed'] else 'FAIL'}] "
            f"{item['name']}: {item['evidence']}"
        )
    return 0 if passed == len(checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
