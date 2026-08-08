from __future__ import annotations

import threading
import time
from pathlib import Path

from app.agentic.models import (
    ModelOutput,
    ModelUsage,
    ResearchBudget,
    RunStatus,
)
from app.agentic.repository import AgenticRunRepository
from app.agentic.runtime import (
    DeepResearchService,
    DeepSeekResearchModel,
    LocalSearchTool,
)
from app.core.bm25 import BM25Index
from app.core.llm import LLMResult, LLMUsage
from app.models import Document


class FakeResearchModel:
    model_name = "fake-research-model"

    def __init__(self, *, clarify: bool = False) -> None:
        self.clarify = clarify
        self.stages: list[str] = []

    def invoke(self, stage: str, payload: dict[str, object]) -> ModelOutput:
        self.stages.append(stage)
        outputs = {
            "scope": {
                "needs_clarification": self.clarify,
                "clarification_question": (
                    "Which time range?" if self.clarify else None
                ),
                "normalized_query": payload.get("query"),
            },
            "brief": {"research_brief": "Compare the two retrieval signals."},
            "plan": {
                "subtasks": [
                    "Find lexical retrieval evidence",
                    "Find semantic retrieval evidence",
                ]
            },
            "report": {"final_report": "The saved evidence supports a comparison."},
        }
        return ModelOutput(
            data=outputs[stage],
            model=self.model_name,
            usage=ModelUsage(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )


class FakeSearchTool:
    name = "fake_search"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []
        self._lock = threading.Lock()

    def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        query = str(arguments["query"])
        with self._lock:
            self.queries.append(query)
        time.sleep(0.01)
        if self.fail:
            raise TimeoutError("provider timed out")
        return {"query": query, "hits": [{"title": query, "content": "evidence"}]}


def _service(
    tmp_path: Path,
    *,
    model: FakeResearchModel | None = None,
    tool: FakeSearchTool | None = None,
    budget: ResearchBudget | None = None,
) -> DeepResearchService:
    return DeepResearchService(
        model=model or FakeResearchModel(),
        tool=tool or FakeSearchTool(),
        repository=AgenticRunRepository(tmp_path / "agentic.db"),
        default_budget=budget or ResearchBudget(),
    )


def test_deep_research_run_persists_complete_trace(tmp_path: Path) -> None:
    model = FakeResearchModel()
    tool = FakeSearchTool()
    service = _service(tmp_path, model=model, tool=tool)

    run = service.run("How do lexical and semantic retrieval differ?")

    assert run.status == RunStatus.COMPLETED
    assert run.research_brief
    assert len(run.subtasks) == 2
    assert len(run.tool_calls) == 2
    assert all(call.status == "succeeded" for call in run.tool_calls)
    assert run.final_report
    assert run.usage.search_calls == 2
    assert run.usage.agent_steps == 4
    assert run.usage.total_tokens == 20
    assert service.get(run.run_id) == run
    assert model.stages == ["scope", "brief", "plan", "report"]
    assert sorted(tool.queries) == sorted(run.plan)


def test_clarification_pauses_without_search_or_report(tmp_path: Path) -> None:
    service = _service(tmp_path, model=FakeResearchModel(clarify=True))

    run = service.run("Compare it.")

    assert run.status == RunStatus.AWAITING_CLARIFICATION
    assert run.clarification_question == "Which time range?"
    assert run.tool_calls == []
    assert run.final_report is None


def test_search_budget_stops_before_parallel_tools(tmp_path: Path) -> None:
    tool = FakeSearchTool()
    service = _service(
        tmp_path,
        tool=tool,
        budget=ResearchBudget(max_search_calls=1),
    )

    run = service.run("Compare retrieval.")

    assert run.status == RunStatus.BUDGET_EXCEEDED
    assert run.stop_reason == "max_search_calls"
    assert tool.queries == []
    assert run.final_report is None


def test_tool_failure_is_structured_and_does_not_create_report(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, tool=FakeSearchTool(fail=True))

    run = service.run("Compare retrieval.")

    assert run.status == RunStatus.FAILED
    assert run.final_report is None
    assert len(run.tool_calls) == 2
    assert all(call.status == "failed" for call in run.tool_calls)
    assert any(error.code == "tool_execution_failed" for error in run.errors)


class RetryClient:
    model_name = "fake-deepseek"

    def __init__(self) -> None:
        self.calls = 0

    def generate_messages(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResult:
        self.calls += 1
        text = "not json" if self.calls == 1 else '{"research_brief": "valid"}'
        return LLMResult(
            text=text,
            model=self.model_name,
            usage=LLMUsage(
                prompt_tokens=2,
                completion_tokens=1,
                total_tokens=3,
            ),
        )


def test_deepseek_structured_model_retries_and_counts_usage() -> None:
    client = RetryClient()
    model = DeepSeekResearchModel(client)  # type: ignore[arg-type]

    output = model.invoke("brief", {"query": "test"})

    assert client.calls == 2
    assert output.data == {"research_brief": "valid"}
    assert output.usage.total_tokens == 6


def test_local_search_validates_and_truncates_output() -> None:
    index = BM25Index()
    index.build(
        [
            Document(
                id="doc-1",
                title="Evidence",
                content="evidence " * 100,
            )
        ]
    )
    tool = LocalSearchTool(index, max_chars_per_hit=20)

    output = tool.execute({"query": "evidence"})

    assert output["hits"][0]["content_truncated"] is True
    assert len(output["hits"][0]["content"]) == 20


class FakeWebEvidenceTool(FakeSearchTool):
    name = "web_evidence_search"
    supports_page_reads = True

    def __init__(self) -> None:
        super().__init__()
        self.page_read_quotas: list[int] = []

    def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        quota = int(arguments["max_page_reads"])
        with self._lock:
            self.queries.append(str(arguments["query"]))
            self.page_read_quotas.append(quota)
        return {
            "evidence": [],
            "_usage": {
                "page_reads": quota,
                "cache_hits": 1,
            },
        }


def test_page_read_budget_is_partitioned_before_parallel_tools(
    tmp_path: Path,
) -> None:
    tool = FakeWebEvidenceTool()
    service = _service(
        tmp_path,
        tool=tool,  # type: ignore[arg-type]
        budget=ResearchBudget(max_page_reads=3),
    )

    run = service.run("Compare retrieval.")

    assert run.status == RunStatus.COMPLETED
    assert sorted(tool.page_read_quotas) == [1, 2]
    assert run.usage.page_reads == 3
    assert run.usage.page_cache_hits == 2
