from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.agentic.api import get_deep_research_service
from app.agentic.models import ModelOutput, ResearchBudget
from app.agentic.repository import AgenticRunRepository
from app.agentic.runtime import DeepResearchService
from app.main import app


class ApiFakeModel:
    model_name = "fake-api-model"

    def invoke(self, stage: str, payload: dict[str, object]) -> ModelOutput:
        outputs = {
            "scope": {
                "needs_clarification": False,
                "clarification_question": None,
                "normalized_query": payload.get("query"),
            },
            "brief": {"research_brief": "A fixed API brief."},
            "plan": {"subtasks": ["Find one source"]},
            "report": {"final_report": "A fixed grounded report."},
        }
        return ModelOutput(data=outputs[stage], model=self.model_name)


class ApiFakeTool:
    name = "fake_search"

    def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"query": arguments["query"], "hits": [{"id": "source-1"}]}


def test_v2_api_creates_and_reads_persisted_run(tmp_path: Path) -> None:
    service = DeepResearchService(
        model=ApiFakeModel(),
        tool=ApiFakeTool(),
        repository=AgenticRunRepository(tmp_path / "api.db"),
        default_budget=ResearchBudget(
            max_wall_time_seconds=30,
            max_agent_steps=10,
            max_search_calls=5,
            max_total_tokens=100,
            max_parallel_research_units=2,
        ),
    )
    app.dependency_overrides[get_deep_research_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/api/v2/research/runs",
            json={"query": "Explain hybrid retrieval."},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["final_report"] == "A fixed grounded report."
        assert payload["tool_calls"][0]["tool_name"] == "fake_search"

        loaded = client.get(
            f"/api/v2/research/runs/{payload['run_id']}"
        )
        assert loaded.status_code == 200
        assert loaded.json()["run_id"] == payload["run_id"]
    finally:
        app.dependency_overrides.clear()


def test_v2_api_is_explicitly_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DEEPTRACE_ENABLED", raising=False)
    client = TestClient(app)

    response = client.post(
        "/api/v2/research/runs",
        json={"query": "Explain hybrid retrieval."},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "deeptrace_v2_disabled"


def test_v2_status_reports_capabilities_without_exposing_secrets(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPTRACE_ENABLED", "true")
    monkeypatch.setenv("DEEPTRACE_WORKFLOW", "supervisor")
    monkeypatch.setenv("DEEPTRACE_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-model-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-search-key")
    client = TestClient(app)

    response = client.get("/api/v2/research/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["workflow"] == "supervisor"
    assert payload["search_provider"] == "brave"
    assert payload["supports_cancel_resume"] is True
    assert payload["limits"]["max_agent_steps"] == 40
    assert "secret-model-key" not in response.text
    assert "secret-search-key" not in response.text


def test_v2_api_reports_missing_deepseek_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTRACE_ENABLED", "true")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = TestClient(app)

    response = client.post(
        "/api/v2/research/runs",
        json={"query": "Explain hybrid retrieval."},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_not_configured"


def test_v2_api_reports_missing_brave_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTRACE_ENABLED", "true")
    monkeypatch.setenv("DEEPTRACE_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    client = TestClient(app)

    response = client.post(
        "/api/v2/research/runs",
        json={"query": "Explain hybrid retrieval."},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "web_search_not_configured"


def test_cancel_requires_supervisor_workflow(tmp_path: Path) -> None:
    service = DeepResearchService(
        model=ApiFakeModel(),
        tool=ApiFakeTool(),
        repository=AgenticRunRepository(tmp_path / "api.db"),
        default_budget=ResearchBudget(),
    )
    app.dependency_overrides[get_deep_research_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post("/api/v2/research/runs/missing/cancel")
        assert response.status_code == 409
        assert (
            response.json()["detail"]["code"]
            == "supervisor_workflow_required"
        )
    finally:
        app.dependency_overrides.clear()
