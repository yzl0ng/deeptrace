from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    QUEUED = "queued"
    SCOPING = "scoping"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PLANNING = "planning"
    RESEARCHING = "researching"
    CHECKING_EVIDENCE = "checking_evidence"
    COMPRESSING = "compressing"
    WRITING = "writing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class ResearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_wall_time_seconds: int = Field(default=900, ge=1)
    max_agent_steps: int = Field(default=40, ge=1)
    max_search_calls: int = Field(default=20, ge=0)
    max_page_reads: int = Field(default=20, ge=0)
    max_total_tokens: int = Field(default=120_000, ge=1)
    max_parallel_research_units: int = Field(default=4, ge=1)


class RunUsage(BaseModel):
    agent_steps: int = 0
    search_calls: int = 0
    page_reads: int = 0
    page_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunError(BaseModel):
    code: str
    message: str
    stage: str


class ToolCallRecord(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    status: str
    observation: dict[str, Any]
    error: RunError | None = None
    token_usage: int | None = None
    cost: float | None = None


class ResearchSubtask(BaseModel):
    subtask_id: str
    question: str
    status: str = "pending"
    tool_call_ids: list[str] = Field(default_factory=list)
    finding: str | None = None
    error: RunError | None = None


class EvidenceAssessment(BaseModel):
    subtask_id: str
    sufficient: bool
    reason: str
    missing_questions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)


class QueryRewrite(BaseModel):
    subtask_id: str
    original_query: str
    rewritten_query: str
    reason: str
    attempt: int = Field(default=1, ge=1)


class ContradictionRecord(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: str = "material"
    explanation: str


class MemoryFold(BaseModel):
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    original_character_count: int = Field(default=0, ge=0)
    folded_character_count: int = Field(default=0, ge=0)


class RunCheckpoint(BaseModel):
    version: int = Field(default=0, ge=0)
    stage: str = "queued"
    completed_stages: list[str] = Field(default_factory=list)
    saved_at: datetime = Field(default_factory=utc_now)


class ResearchRun(BaseModel):
    run_id: str
    user_query: str
    mode: str = "deep_research_baseline"
    status: RunStatus = RunStatus.QUEUED
    clarification_question: str | None = None
    research_brief: str | None = None
    plan: list[str] = Field(default_factory=list)
    subtasks: list[ResearchSubtask] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    evidence_assessments: list[EvidenceAssessment] = Field(
        default_factory=list
    )
    query_rewrites: list[QueryRewrite] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    memory: MemoryFold | None = None
    checkpoint: RunCheckpoint = Field(default_factory=RunCheckpoint)
    cancel_requested: bool = False
    budget: ResearchBudget
    usage: RunUsage = Field(default_factory=RunUsage)
    stop_reason: str | None = None
    errors: list[RunError] = Field(default_factory=list)
    final_report: str | None = None
    final_evidence_ids: list[str] = Field(default_factory=list)
    model_name: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateResearchRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    budget: ResearchBudget | None = None


class ModelUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelOutput(BaseModel):
    data: dict[str, Any]
    model: str
    usage: ModelUsage = Field(default_factory=ModelUsage)
