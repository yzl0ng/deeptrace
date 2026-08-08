from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.agentic.models import (
    ModelOutput,
    ModelUsage,
    ResearchBudget,
    RunStatus,
)
from app.agentic.repository import AgenticRunRepository
from app.agentic.supervisor import (
    SupervisorResearchService,
    _bounded_observation,
)


class SupervisorFakeModel:
    model_name = "fake-supervisor"

    def __init__(self) -> None:
        self.stages: list[str] = []

    def invoke(self, stage: str, payload: dict[str, object]) -> ModelOutput:
        self.stages.append(stage)
        if stage == "scope":
            data = {
                "needs_clarification": False,
                "clarification_question": None,
                "normalized_query": payload["query"],
            }
        elif stage == "brief":
            data = {
                "research_brief": (
                    "Compare the retrieval methods and explain fusion."
                )
            }
        elif stage == "plan":
            data = {
                "subtasks": [
                    "What does BM25 optimize?",
                    "How do dense retrieval and RRF differ?",
                ]
            }
        elif stage == "evidence_grade":
            observations = payload["observations"]
            question = str(payload["question"])
            insufficient = "BM25" in question and len(observations) == 1
            data = {
                "sufficient": not insufficient,
                "reason": (
                    "Missing an authoritative scoring explanation."
                    if insufficient
                    else "The evidence directly answers the subtask."
                ),
                "missing_questions": (
                    ["Find the BM25 scoring components."]
                    if insufficient
                    else []
                ),
            }
        elif stage == "query_rewrite":
            data = {
                "rewritten_query": "official BM25 scoring components",
                "reason": "Target the missing scoring details.",
            }
        elif stage == "contradictions":
            evidence = payload["evidence"]
            identifiers = [
                identifier
                for item in evidence
                for identifier in item["evidence_ids"]
            ]
            data = {
                "contradictions": [
                    {
                        "claim": "Raw BM25 and cosine scores are comparable.",
                        "evidence_ids": identifiers[:2],
                        "severity": "material",
                        "explanation": (
                            "The sources describe different score spaces."
                        ),
                    }
                ]
            }
        elif stage == "memory_fold":
            data = {
                "summary": (
                    "BM25 is lexical, dense retrieval is semantic, and "
                    "RRF combines ranks while retaining source IDs."
                )
            }
        elif stage == "report":
            evidence_ids = payload["allowed_evidence_ids"]
            data = {
                "final_report": (
                    "Use BM25 and dense retrieval as complementary "
                    "signals, then fuse their ranks with RRF."
                ),
                "cited_evidence_ids": evidence_ids[:2],
            }
        else:
            raise AssertionError(stage)
        return ModelOutput(
            data=data,
            model=self.model_name,
            usage=ModelUsage(
                prompt_tokens=2,
                completion_tokens=1,
                total_tokens=3,
            ),
        )


class ConcurrentEvidenceTool:
    name = "fake_web_evidence"
    supports_page_reads = True

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        query = str(arguments["query"])
        with self._lock:
            self.queries.append(query)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        evidence_id = "ev-" + str(len(self.queries)) + "-" + str(
            abs(hash(query)) % 10_000
        )
        return {
            "query": query,
            "evidence": [
                {
                    "evidence_id": evidence_id,
                    "content": f"Evidence for {query}",
                }
            ],
            "_usage": {"page_reads": 1, "cache_hits": 0},
        }


def _service(
    tmp_path: Path,
    *,
    model: SupervisorFakeModel | None = None,
    tool: ConcurrentEvidenceTool | None = None,
    checkpoint_hook=None,
) -> SupervisorResearchService:
    return SupervisorResearchService(
        model=model or SupervisorFakeModel(),
        tool=tool or ConcurrentEvidenceTool(),
        repository=AgenticRunRepository(tmp_path / "supervisor.db"),
        default_budget=ResearchBudget(
            max_agent_steps=20,
            max_search_calls=6,
            max_page_reads=6,
            max_parallel_research_units=2,
            max_total_tokens=1000,
        ),
        max_rewrite_attempts=1,
        checkpoint_hook=checkpoint_hook,
    )


def test_supervisor_rewrites_insufficient_evidence_and_folds_trace(
    tmp_path: Path,
) -> None:
    model = SupervisorFakeModel()
    tool = ConcurrentEvidenceTool()
    service = _service(tmp_path, model=model, tool=tool)

    run = service.run(
        "Why should BM25 and dense scores use rank fusion?"
    )

    assert run.status == RunStatus.COMPLETED
    assert len(run.subtasks) == 2
    assert tool.max_active == 2
    assert tool.max_active <= run.budget.max_parallel_research_units
    assert run.usage.search_calls == 3
    assert run.usage.page_reads == 3
    assert len(run.query_rewrites) == 1
    assert run.query_rewrites[0].rewritten_query.startswith("official")
    assert [item.sufficient for item in run.evidence_assessments] == [
        False,
        True,
        True,
    ]
    assert len(run.contradictions) == 1
    assert run.memory is not None
    assert run.memory.evidence_ids
    assert len(run.memory.tool_call_ids) == 3
    assert run.memory.folded_character_count < (
        run.memory.original_character_count
    )
    assert run.checkpoint.completed_stages[-4:] == [
        "evidence",
        "contradictions",
        "memory",
        "report",
    ]
    assert {
        "evidence:subtask-1",
        "evidence:subtask-2",
    }.issubset(run.checkpoint.completed_stages)
    assert run.final_report
    assert run.final_evidence_ids
    assert set(run.final_evidence_ids).issubset(
        set(run.memory.evidence_ids)
    )


class AlwaysInsufficientModel(SupervisorFakeModel):
    def invoke(self, stage: str, payload: dict[str, object]) -> ModelOutput:
        if stage == "evidence_grade":
            self.stages.append(stage)
            return ModelOutput(
                data={
                    "sufficient": False,
                    "reason": "The comparison target is not covered.",
                    "missing_questions": ["Find the missing comparison."],
                },
                model=self.model_name,
            )
        return super().invoke(stage, payload)


def test_supervisor_refuses_when_evidence_remains_insufficient(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, model=AlwaysInsufficientModel())

    run = service.run("Compare unsupported systems.")

    assert run.status == RunStatus.COMPLETED
    assert run.stop_reason == "insufficient_evidence"
    assert run.final_report.startswith("Evidence is insufficient")
    assert run.final_evidence_ids
    assert all(item.status == "insufficient" for item in run.subtasks)
    assert run.memory is None
    assert run.contradictions == []
    assert "report" in run.checkpoint.completed_stages


class SimulatedProcessExit(BaseException):
    pass


def test_resume_uses_checkpoint_without_repeating_completed_research(
    tmp_path: Path,
) -> None:
    crashed_run_id: list[str] = []

    def crash_after_research(run, stage: str) -> None:
        if stage == "research":
            crashed_run_id.append(run.run_id)
            raise SimulatedProcessExit()

    model = SupervisorFakeModel()
    tool = ConcurrentEvidenceTool()
    interrupted = _service(
        tmp_path,
        model=model,
        tool=tool,
        checkpoint_hook=crash_after_research,
    )
    with pytest.raises(SimulatedProcessExit):
        interrupted.run("Compare BM25, dense retrieval, and RRF.")

    assert len(tool.queries) == 2
    repository = AgenticRunRepository(tmp_path / "supervisor.db")
    saved = repository.get(crashed_run_id[0])
    assert saved is not None
    assert "research" in saved.checkpoint.completed_stages

    resumed = _service(tmp_path, model=model, tool=tool).resume(
        crashed_run_id[0]
    )
    assert resumed is not None
    assert resumed.status == RunStatus.COMPLETED
    assert len(tool.queries) == 3
    assert resumed.usage.search_calls == 3
    assert resumed.checkpoint.version > saved.checkpoint.version


def test_cancelled_checkpoint_can_be_resumed(tmp_path: Path) -> None:
    run_id: list[str] = []

    def interrupt_after_plan(run, stage: str) -> None:
        if stage == "plan":
            run_id.append(run.run_id)
            raise SimulatedProcessExit()

    service = _service(tmp_path, checkpoint_hook=interrupt_after_plan)
    with pytest.raises(SimulatedProcessExit):
        service.run("Compare BM25, dense retrieval, and RRF.")

    cancelled = service.cancel(run_id[0])
    assert cancelled is not None
    assert cancelled.status == RunStatus.CANCELLED
    assert cancelled.cancel_requested is True

    resumed = _service(tmp_path).resume(run_id[0])
    assert resumed is not None
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.cancel_requested is False


def test_supervisor_rejects_single_subtask_plan(tmp_path: Path) -> None:
    class SingleTaskModel(SupervisorFakeModel):
        def invoke(
            self, stage: str, payload: dict[str, object]
        ) -> ModelOutput:
            output = super().invoke(stage, payload)
            if stage == "plan":
                output.data = {"subtasks": ["Only one task"]}
            return output

    run = _service(tmp_path, model=SingleTaskModel()).run(
        "A multi-hop research question"
    )
    assert run.status == RunStatus.FAILED
    assert "at least two meaningful subtasks" in run.errors[-1].message


def test_context_bounding_preserves_evidence_trace() -> None:
    bounded = _bounded_observation(
        {
            "query": "large page",
            "evidence": [
                {
                    "evidence_id": "ev-large",
                    "source_id": "src-large",
                    "passage_id": "psg-large",
                    "content": "x" * 20_000,
                }
            ],
        },
        1000,
    )

    assert bounded["_context_truncated"] is True
    assert bounded["_content_sha256"]
    assert bounded["evidence"][0]["evidence_id"] == "ev-large"
    assert len(bounded["evidence"][0].get("content", "")) < 20_000
