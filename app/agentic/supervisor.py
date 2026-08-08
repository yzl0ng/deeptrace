from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.agentic.models import (
    ContradictionRecord,
    EvidenceAssessment,
    MemoryFold,
    ModelUsage,
    QueryRewrite,
    ResearchBudget,
    ResearchRun,
    ResearchSubtask,
    RunCheckpoint,
    RunError,
    RunStatus,
    ToolCallRecord,
    utc_now,
)
from app.agentic.repository import AgenticRunRepository
from app.agentic.runtime import (
    BudgetExceeded,
    ResearchModel,
    ResearchTool,
    StructuredModelError,
)


class WorkflowCancelled(RuntimeError):
    pass


CheckpointHook = Callable[[ResearchRun, str], None]


class SupervisorResearchService:
    """Checkpointed Phase-3 workflow with bounded query repair."""

    def __init__(
        self,
        *,
        model: ResearchModel,
        tool: ResearchTool,
        repository: AgenticRunRepository,
        default_budget: ResearchBudget,
        max_rewrite_attempts: int = 1,
        max_context_chars: int = 128_000,
        clock: Any = time.monotonic,
        checkpoint_hook: CheckpointHook | None = None,
    ) -> None:
        if max_rewrite_attempts < 0:
            raise ValueError("max_rewrite_attempts must be non-negative")
        if max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
        self.model = model
        self.tool = tool
        self.repository = repository
        self.default_budget = default_budget
        self.max_rewrite_attempts = max_rewrite_attempts
        self.max_context_chars = max_context_chars
        self.clock = clock
        self.checkpoint_hook = checkpoint_hook

    def run(
        self,
        query: str,
        *,
        budget: ResearchBudget | None = None,
    ) -> ResearchRun:
        run = ResearchRun(
            run_id=f"run-{uuid4().hex}",
            user_query=query.strip(),
            mode="deep_research_supervisor",
            budget=budget or self.default_budget,
            model_name=self.model.model_name,
        )
        self.repository.save(run)
        return self._execute(run)

    def get(self, run_id: str) -> ResearchRun | None:
        return self.repository.get(run_id)

    def cancel(self, run_id: str) -> ResearchRun | None:
        return self.repository.request_cancel(run_id)

    def resume(self, run_id: str) -> ResearchRun | None:
        run = self.repository.get(run_id)
        if run is None:
            return None
        if run.status == RunStatus.COMPLETED:
            return run
        run.cancel_requested = False
        run.stop_reason = None
        run.errors = [
            error
            for error in run.errors
            if error.code not in {"workflow_cancelled", "workflow_interrupted"}
        ]
        self.repository.save(run)
        return self._execute(run)

    def _execute(self, run: ResearchRun) -> ResearchRun:
        started = self.clock()
        try:
            if not self._done(run, "scope"):
                scope = self._model_step(
                    run,
                    started,
                    RunStatus.SCOPING,
                    "scope",
                    {"query": run.user_query},
                )
                if bool(scope.get("needs_clarification", False)):
                    run.status = RunStatus.AWAITING_CLARIFICATION
                    run.clarification_question = str(
                        scope.get("clarification_question")
                        or "Please clarify the research scope."
                    )
                    run.stop_reason = "clarification_required"
                    return self._checkpoint(run, "scope")
                self._checkpoint(run, "scope")

            if not self._done(run, "brief"):
                brief = self._model_step(
                    run,
                    started,
                    RunStatus.SCOPING,
                    "brief",
                    {"query": run.user_query},
                )
                run.research_brief = _required_text(
                    brief, "research_brief"
                )
                self._checkpoint(run, "brief")

            if not self._done(run, "plan"):
                planned = self._model_step(
                    run,
                    started,
                    RunStatus.PLANNING,
                    "plan",
                    {
                        "query": run.user_query,
                        "research_brief": run.research_brief,
                        "max_subtasks": (
                            run.budget.max_parallel_research_units
                        ),
                        "require_multiple_for_multihop": True,
                    },
                )
                questions = _subtask_questions(
                    planned,
                    run.budget.max_parallel_research_units,
                )
                run.plan = questions
                run.subtasks = [
                    ResearchSubtask(
                        subtask_id=f"subtask-{index + 1}",
                        question=question,
                    )
                    for index, question in enumerate(questions)
                ]
                self._checkpoint(run, "plan")

            if not self._done(run, "research"):
                self._research_pending_subtasks(run, started)
                self._checkpoint(run, "research")

            if not self._done(run, "evidence"):
                self._grade_and_rewrite(run, started)
                self._checkpoint(run, "evidence")

            insufficient = [
                subtask
                for subtask in run.subtasks
                if subtask.status != "completed"
            ]
            if insufficient:
                missing = [
                    assessment
                    for assessment in _latest_evidence_assessments(run)
                    if not assessment.sufficient
                ]
                details = "; ".join(
                    assessment.reason for assessment in missing
                )
                run.final_report = (
                    "Evidence is insufficient to answer safely. "
                    + (
                        f"Missing support: {details}"
                        if details
                        else "One or more research subtasks remain unsupported."
                    )
                )
                run.final_evidence_ids = self._all_evidence_ids(run)
                run.status = RunStatus.COMPLETED
                run.stop_reason = "insufficient_evidence"
                self._checkpoint(run, "report")
                return run

            if not self._done(run, "contradictions"):
                result = self._model_step(
                    run,
                    started,
                    RunStatus.CHECKING_EVIDENCE,
                    "contradictions",
                    {
                        "query": run.user_query,
                        "evidence": self._evidence_payload(run),
                    },
                )
                run.contradictions = _contradictions(
                    result,
                    allowed_evidence_ids=set(
                        self._all_evidence_ids(run)
                    ),
                )
                self._checkpoint(run, "contradictions")

            if not self._done(run, "memory"):
                original = json.dumps(
                    self._evidence_payload(run, bounded=False),
                    ensure_ascii=False,
                )
                result = self._model_step(
                    run,
                    started,
                    RunStatus.COMPRESSING,
                    "memory_fold",
                    {
                        "query": run.user_query,
                        "evidence": self._evidence_payload(run),
                        "contradictions": [
                            item.model_dump(mode="json")
                            for item in run.contradictions
                        ],
                    },
                )
                evidence_ids = self._all_evidence_ids(run)
                run.memory = MemoryFold(
                    summary=_required_text(result, "summary"),
                    evidence_ids=evidence_ids,
                    tool_call_ids=[
                        call.call_id for call in run.tool_calls
                    ],
                    original_character_count=len(original),
                    folded_character_count=len(
                        _required_text(result, "summary")
                    ),
                )
                self._checkpoint(run, "memory")

            if not self._done(run, "report"):
                result = self._model_step(
                    run,
                    started,
                    RunStatus.WRITING,
                    "report",
                    {
                        "query": run.user_query,
                        "research_brief": run.research_brief,
                        "memory": (
                            run.memory.model_dump(mode="json")
                            if run.memory
                            else None
                        ),
                        "contradictions": [
                            item.model_dump(mode="json")
                            for item in run.contradictions
                        ],
                        "allowed_evidence_ids": self._all_evidence_ids(run),
                        "citation_requirement": (
                            "Return cited_evidence_ids using only the allowed "
                            "IDs. Cite at least one ID for a substantive answer."
                        ),
                    },
                )
                run.final_report = _required_text(
                    result, "final_report"
                )
                run.final_evidence_ids = _report_evidence_ids(
                    result,
                    allowed_evidence_ids=set(
                        self._all_evidence_ids(run)
                    ),
                )
                run.status = RunStatus.COMPLETED
                run.stop_reason = "completed"
                self._checkpoint(run, "report")
            return run
        except WorkflowCancelled as error:
            run.status = RunStatus.CANCELLED
            run.cancel_requested = True
            run.stop_reason = "cancel_requested"
            run.errors.append(
                RunError(
                    code="workflow_cancelled",
                    message=str(error),
                    stage=run.checkpoint.stage,
                )
            )
            self.repository.save(run)
            return run
        except BudgetExceeded as error:
            run.status = RunStatus.BUDGET_EXCEEDED
            run.stop_reason = str(error)
            run.errors.append(
                RunError(
                    code="budget_exceeded",
                    message=str(error),
                    stage=run.checkpoint.stage,
                )
            )
            self.repository.save(run)
            return run
        except Exception as error:
            run.status = RunStatus.FAILED
            run.stop_reason = "structured_failure"
            run.errors.append(
                RunError(
                    code=getattr(
                        error, "code", "supervisor_workflow_failed"
                    ),
                    message=str(error),
                    stage=run.checkpoint.stage,
                )
            )
            self.repository.save(run)
            return run

    def _research_pending_subtasks(
        self,
        run: ResearchRun,
        started: float,
    ) -> None:
        pending = [
            item for item in run.subtasks if item.status != "completed"
        ]
        if not pending:
            return
        self._reserve_search_calls(run, len(pending), started)
        quotas = self._page_read_quotas(run, len(pending))
        max_workers = min(
            len(pending),
            run.budget.max_parallel_research_units,
        )
        run.status = RunStatus.RESEARCHING
        self._save(run)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_tool,
                    item.question,
                    quotas[index],
                ): item
                for index, item in enumerate(pending)
            }
            for future in as_completed(futures):
                self._check_cancel(run)
                subtask = futures[future]
                record = future.result()
                self._record_tool_result(run, subtask, record)
                self._checkpoint(
                    run,
                    f"research:{subtask.subtask_id}",
                    complete=False,
                )
        if any(item.status == "failed" for item in pending):
            raise RuntimeError("one or more research subtasks failed")

    def _grade_and_rewrite(
        self,
        run: ResearchRun,
        started: float,
    ) -> None:
        for subtask in run.subtasks:
            subtask_stage = f"evidence:{subtask.subtask_id}"
            if self._done(run, subtask_stage):
                continue
            self._check_cancel(run)
            attempt = 1 + sum(
                1
                for item in run.evidence_assessments
                if item.subtask_id == subtask.subtask_id
            )
            assessment = self._grade_subtask(
                run, subtask, attempt, started
            )
            run.evidence_assessments.append(assessment)
            self._save(run)
            while (
                not assessment.sufficient
                and attempt <= self.max_rewrite_attempts
            ):
                rewritten = self._model_step(
                    run,
                    started,
                    RunStatus.CHECKING_EVIDENCE,
                    "query_rewrite",
                    {
                        "query": subtask.question,
                        "reason": assessment.reason,
                        "missing_questions": (
                            assessment.missing_questions
                        ),
                        "attempt": attempt,
                    },
                )
                new_query = _required_text(
                    rewritten, "rewritten_query"
                )
                run.query_rewrites.append(
                    QueryRewrite(
                        subtask_id=subtask.subtask_id,
                        original_query=subtask.question,
                        rewritten_query=new_query,
                        reason=str(
                            rewritten.get("reason")
                            or assessment.reason
                        ),
                        attempt=attempt,
                    )
                )
                self._reserve_search_calls(run, 1, started)
                quota = self._page_read_quotas(run, 1)[0]
                record = self._execute_tool(new_query, quota)
                self._record_tool_result(run, subtask, record)
                if record.status != "succeeded":
                    raise RuntimeError("rewritten research query failed")
                attempt += 1
                assessment = self._grade_subtask(
                    run, subtask, attempt, started
                )
                run.evidence_assessments.append(assessment)
                self._save(run)
            subtask.status = (
                "completed" if assessment.sufficient else "insufficient"
            )
            self._checkpoint(run, subtask_stage)

    def _grade_subtask(
        self,
        run: ResearchRun,
        subtask: ResearchSubtask,
        attempt: int,
        started: float,
    ) -> EvidenceAssessment:
        result = self._model_step(
            run,
            started,
            RunStatus.CHECKING_EVIDENCE,
            "evidence_grade",
            {
                "question": subtask.question,
                "observations": self._subtask_observations(
                    run,
                    subtask,
                    bounded=True,
                ),
                "attempt": attempt,
            },
        )
        return EvidenceAssessment(
            subtask_id=subtask.subtask_id,
            sufficient=bool(result.get("sufficient", False)),
            reason=str(result.get("reason") or "No reason supplied."),
            missing_questions=[
                str(item)
                for item in result.get("missing_questions", [])
                if str(item).strip()
            ],
            evidence_ids=self._subtask_evidence_ids(run, subtask),
            attempt=attempt,
        )

    def _execute_tool(
        self,
        query: str,
        page_read_quota: int | None,
    ) -> ToolCallRecord:
        call_id = f"call-{uuid4().hex}"
        arguments: dict[str, Any] = {"query": query}
        if page_read_quota is not None:
            arguments["max_page_reads"] = page_read_quota
        started_at = datetime.now(UTC)
        try:
            observation = self.tool.execute(arguments)
            status = "succeeded"
            error = None
        except Exception as exc:
            observation = {}
            status = "failed"
            error = RunError(
                code="tool_execution_failed",
                message=str(exc),
                stage="researching",
            )
        return ToolCallRecord(
            call_id=call_id,
            tool_name=self.tool.name,
            arguments=arguments,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status=status,
            observation=observation,
            error=error,
        )

    def _record_tool_result(
        self,
        run: ResearchRun,
        subtask: ResearchSubtask,
        record: ToolCallRecord,
    ) -> None:
        run.tool_calls.append(record)
        subtask.tool_call_ids.append(record.call_id)
        usage = record.observation.get("_usage", {})
        run.usage.page_reads += int(usage.get("page_reads", 0))
        run.usage.page_cache_hits += int(usage.get("cache_hits", 0))
        if record.error is not None:
            subtask.status = "failed"
            subtask.error = record.error
            run.errors.append(record.error)
            subtask.finding = None
        else:
            subtask.status = "completed"
            observations = self._subtask_observations(
                run,
                subtask,
                bounded=True,
            )
            subtask.finding = json.dumps(
                observations, ensure_ascii=False
            )
        self._save(run)

    def _model_step(
        self,
        run: ResearchRun,
        started: float,
        status: RunStatus,
        stage: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._check_cancel(run)
        run.status = status
        run.usage.agent_steps += 1
        self._check_budget(run, started)
        self._save(run)
        try:
            output = self.model.invoke(stage, payload)
        except StructuredModelError as error:
            self._add_usage(run, error.usage)
            self._check_budget(run, started)
            raise
        run.model_name = output.model
        self._add_usage(run, output.usage)
        self._check_budget(run, started)
        return output.data

    def _checkpoint(
        self,
        run: ResearchRun,
        stage: str,
        *,
        complete: bool = True,
    ) -> ResearchRun:
        self._check_cancel(run)
        completed = list(run.checkpoint.completed_stages)
        if complete and stage not in completed:
            completed.append(stage)
        run.checkpoint = RunCheckpoint(
            version=run.checkpoint.version + 1,
            stage=stage,
            completed_stages=completed,
        )
        self._save(run)
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(run, stage)
        return run

    def _save(self, run: ResearchRun) -> None:
        persisted = self.repository.get(run.run_id)
        if persisted is not None and persisted.cancel_requested:
            run.cancel_requested = True
        run.updated_at = utc_now()
        self.repository.save(run)

    def _check_cancel(self, run: ResearchRun) -> None:
        persisted = self.repository.get(run.run_id)
        if run.cancel_requested or (
            persisted is not None and persisted.cancel_requested
        ):
            raise WorkflowCancelled("research run cancellation requested")

    def _check_budget(self, run: ResearchRun, started: float) -> None:
        if self.clock() - started > run.budget.max_wall_time_seconds:
            raise BudgetExceeded("max_wall_time_seconds")
        if run.usage.agent_steps > run.budget.max_agent_steps:
            raise BudgetExceeded("max_agent_steps")
        if run.usage.search_calls > run.budget.max_search_calls:
            raise BudgetExceeded("max_search_calls")
        if run.usage.page_reads > run.budget.max_page_reads:
            raise BudgetExceeded("max_page_reads")
        if run.usage.total_tokens > run.budget.max_total_tokens:
            raise BudgetExceeded("max_total_tokens")

    def _reserve_search_calls(
        self,
        run: ResearchRun,
        count: int,
        started: float,
    ) -> None:
        if run.usage.search_calls + count > run.budget.max_search_calls:
            raise BudgetExceeded("max_search_calls")
        run.usage.search_calls += count
        self._check_budget(run, started)
        self._save(run)

    def _page_read_quotas(
        self,
        run: ResearchRun,
        count: int,
    ) -> list[int | None]:
        if not getattr(self.tool, "supports_page_reads", False):
            return [None] * count
        remaining = max(
            run.budget.max_page_reads - run.usage.page_reads,
            0,
        )
        quotas: list[int | None] = []
        for index in range(count):
            tasks_left = count - index
            quota = remaining // tasks_left
            quotas.append(quota)
            remaining -= quota
        return quotas

    def _subtask_observations(
        self,
        run: ResearchRun,
        subtask: ResearchSubtask,
        *,
        bounded: bool = False,
    ) -> list[dict[str, Any]]:
        call_ids = set(subtask.tool_call_ids)
        observations = [
            call.observation
            for call in run.tool_calls
            if call.call_id in call_ids and call.status == "succeeded"
        ]
        if not bounded:
            return observations
        per_observation = max(
            self.max_context_chars // max(len(observations), 1),
            1000,
        )
        return [
            _bounded_observation(item, per_observation)
            for item in observations
        ]

    def _subtask_evidence_ids(
        self,
        run: ResearchRun,
        subtask: ResearchSubtask,
    ) -> list[str]:
        identifiers: list[str] = []
        for observation in self._subtask_observations(
            run, subtask, bounded=False
        ):
            for item in observation.get("evidence", []):
                if isinstance(item, dict) and item.get("evidence_id"):
                    identifiers.append(str(item["evidence_id"]))
        if identifiers:
            return list(dict.fromkeys(identifiers))
        return [f"tool:{call_id}" for call_id in subtask.tool_call_ids]

    def _all_evidence_ids(self, run: ResearchRun) -> list[str]:
        identifiers: list[str] = []
        for subtask in run.subtasks:
            identifiers.extend(
                self._subtask_evidence_ids(run, subtask)
            )
        return list(dict.fromkeys(identifiers))

    def _evidence_payload(
        self,
        run: ResearchRun,
        *,
        bounded: bool = True,
    ) -> list[dict[str, Any]]:
        per_subtask = max(
            self.max_context_chars // max(len(run.subtasks), 1),
            1000,
        )
        return [
            {
                "subtask_id": subtask.subtask_id,
                "question": subtask.question,
                "evidence_ids": self._subtask_evidence_ids(run, subtask),
                "observations": (
                    [
                        _bounded_observation(item, per_subtask)
                        for item in self._subtask_observations(
                            run,
                            subtask,
                            bounded=False,
                        )
                    ]
                    if bounded
                    else self._subtask_observations(
                        run,
                        subtask,
                        bounded=False,
                    )
                ),
            }
            for subtask in run.subtasks
        ]

    @staticmethod
    def _add_usage(run: ResearchRun, usage: ModelUsage) -> None:
        run.usage.prompt_tokens += usage.prompt_tokens
        run.usage.completion_tokens += usage.completion_tokens
        run.usage.total_tokens += usage.total_tokens

    @staticmethod
    def _done(run: ResearchRun, stage: str) -> bool:
        return stage in run.checkpoint.completed_stages


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StructuredModelError(f"{key} must be a non-empty string")
    return value.strip()


def _subtask_questions(
    data: dict[str, Any],
    maximum: int,
) -> list[str]:
    raw = data.get("subtasks")
    if not isinstance(raw, list):
        raise StructuredModelError("plan.subtasks must be a list")
    questions = [
        str(item).strip() for item in raw[:maximum] if str(item).strip()
    ]
    if len(questions) < 2:
        raise StructuredModelError(
            "supervisor plan must contain at least two meaningful subtasks"
        )
    return questions


def _contradictions(
    data: dict[str, Any],
    *,
    allowed_evidence_ids: set[str],
) -> list[ContradictionRecord]:
    raw = data.get("contradictions", [])
    if not isinstance(raw, list):
        raise StructuredModelError("contradictions must be a list")
    records = [
        ContradictionRecord.model_validate(item)
        for item in raw
        if isinstance(item, dict)
    ]
    invalid = {
        evidence_id
        for record in records
        for evidence_id in record.evidence_ids
        if evidence_id not in allowed_evidence_ids
    }
    if invalid:
        raise StructuredModelError(
            "contradictions referenced unknown evidence IDs"
        )
    return records


def _latest_evidence_assessments(
    run: ResearchRun,
) -> list[EvidenceAssessment]:
    latest: dict[str, EvidenceAssessment] = {}
    for assessment in run.evidence_assessments:
        previous = latest.get(assessment.subtask_id)
        if previous is None or assessment.attempt >= previous.attempt:
            latest[assessment.subtask_id] = assessment
    return list(latest.values())


def _report_evidence_ids(
    data: dict[str, Any],
    *,
    allowed_evidence_ids: set[str],
) -> list[str]:
    raw = data.get("cited_evidence_ids")
    if not isinstance(raw, list):
        raise StructuredModelError(
            "report.cited_evidence_ids must be a list"
        )
    identifiers = list(
        dict.fromkeys(
            str(item).strip() for item in raw if str(item).strip()
        )
    )
    if not identifiers:
        raise StructuredModelError(
            "report must cite at least one evidence ID"
        )
    invalid = set(identifiers) - allowed_evidence_ids
    if invalid:
        raise StructuredModelError(
            "report referenced unknown evidence IDs"
        )
    return identifiers


def _bounded_observation(
    observation: dict[str, Any],
    maximum_characters: int,
) -> dict[str, Any]:
    serialized = json.dumps(observation, ensure_ascii=False)
    if len(serialized) <= maximum_characters:
        return observation

    result: dict[str, Any] = {
        "query": str(observation.get("query", ""))[:500],
        "_context_truncated": True,
        "_original_character_count": len(serialized),
        "_content_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
    }
    evidence = observation.get("evidence", [])
    if isinstance(evidence, list):
        content_limit = max(
            min(
                maximum_characters // max(len(evidence), 1) // 2,
                2000,
            ),
            128,
        )
        result["evidence"] = [
            {
                key: (
                    str(item[key])[:content_limit]
                    if key == "content"
                    else item[key]
                )
                for key in (
                    "evidence_id",
                    "source_id",
                    "document_id",
                    "passage_id",
                    "canonical_url",
                    "title",
                    "rank",
                    "retrieval_method",
                    "content",
                )
                if key in item
            }
            for item in evidence
            if isinstance(item, dict)
        ]
    hits = observation.get("hits", [])
    if isinstance(hits, list):
        result["hits"] = [
            {
                key: (
                    str(item[key])[:1000]
                    if key == "content"
                    else item[key]
                )
                for key in (
                    "document_id",
                    "title",
                    "source",
                    "rank",
                    "score",
                    "content",
                )
                if key in item
            }
            for item in hits[:10]
            if isinstance(item, dict)
        ]
    failures = observation.get("failures")
    if isinstance(failures, list):
        result["failures"] = failures[:10]

    if len(json.dumps(result, ensure_ascii=False)) > maximum_characters:
        result.pop("hits", None)
        result["evidence"] = [
            {
                key: item[key]
                for key in ("evidence_id", "source_id", "passage_id")
                if key in item
            }
            for item in result.get("evidence", [])
        ]
    return result
