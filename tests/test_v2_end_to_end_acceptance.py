from scripts.run_v2_end_to_end_acceptance import build_acceptance_checks


def test_acceptance_checks_require_grounded_persisted_completion() -> None:
    run = {
        "run_id": "run-1",
        "status": "completed",
        "stop_reason": "completed",
        "model_name": "model",
        "plan": ["one", "two"],
        "subtasks": [
            {"subtask_id": "s1", "status": "completed"},
            {"subtask_id": "s2", "status": "completed"},
        ],
        "tool_calls": [
            {"call_id": "c1", "status": "succeeded"},
            {"call_id": "c2", "status": "succeeded"},
        ],
        "evidence_assessments": [
            {
                "subtask_id": "s1",
                "attempt": 1,
                "sufficient": True,
                "evidence_ids": ["ev-1"],
            },
            {
                "subtask_id": "s2",
                "attempt": 1,
                "sufficient": True,
                "evidence_ids": ["ev-2"],
            },
        ],
        "final_report": "A" * 100,
        "final_evidence_ids": ["ev-1", "ev-2"],
        "checkpoint": {
            "stage": "report",
            "version": 8,
            "completed_stages": ["report"],
        },
        "usage": {
            "agent_steps": 8,
            "search_calls": 2,
            "total_tokens": 100,
        },
        "errors": [],
    }

    checks = build_acceptance_checks(
        status={"ready": True, "workflow": "supervisor"},
        created=run,
        persisted=dict(run),
        web_html="<title>SearchLab</title>Agent 全链路",
    )

    assert all(item["passed"] for item in checks)


def test_acceptance_checks_reject_insufficient_uncited_answer() -> None:
    run = {
        "run_id": "run-2",
        "status": "completed",
        "stop_reason": "insufficient_evidence",
        "plan": ["one", "two"],
        "subtasks": [
            {"subtask_id": "s1", "status": "insufficient"},
        ],
        "tool_calls": [{"call_id": "c1", "status": "succeeded"}],
        "evidence_assessments": [
            {
                "subtask_id": "s1",
                "attempt": 1,
                "sufficient": False,
                "evidence_ids": ["ev-1"],
            }
        ],
        "final_report": "Evidence is insufficient." * 4,
        "final_evidence_ids": [],
        "checkpoint": {
            "stage": "report",
            "version": 5,
            "completed_stages": ["report"],
        },
        "usage": {
            "agent_steps": 5,
            "search_calls": 1,
            "total_tokens": 50,
        },
        "errors": [],
    }

    checks = build_acceptance_checks(
        status={"ready": True, "workflow": "supervisor"},
        created=run,
        persisted=dict(run),
        web_html=None,
    )
    failed = {item["name"] for item in checks if not item["passed"]}

    assert "run_completed" in failed
    assert "subtasks_completed" in failed
    assert "evidence_sufficient" in failed
    assert "evidence_allowlist" in failed
