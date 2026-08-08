from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

ACTION_SYSTEM_PROMPT = (
    "You are the DeepTrace tool-loop policy. Return exactly one JSON object "
    "and no markdown. Schema: {\"rationale_summary\":string,\"action\":"
    "\"search|read_page|evaluate_evidence|answer\",\"arguments\":object,"
    "\"evidence_ids\":[string],\"final_answer\":string|null}. Choose only "
    "the next action. Tool observations will be supplied by the environment; "
    "never invent an observation or evidence ID. Use answer only when the "
    "available evidence is sufficient."
)
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class ToolLoopEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    title: str
    content: str
    source: str


class ToolLoopGoldClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)


class ToolLoopEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset: str
    split: str
    language: str
    question: str
    expected_answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    evidence: list[ToolLoopEvidence]
    gold_claims: list[ToolLoopGoldClaim]
    reviewed: bool
    source_record_id: str


class ToolLoopAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale_summary: str = Field(min_length=3, max_length=500)
    action: Literal["search", "read_page", "evaluate_evidence", "answer"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    final_answer: str | None = Field(default=None, max_length=6000)

    @model_validator(mode="after")
    def validate_answer_fields(self) -> ToolLoopAction:
        if self.action == "answer":
            if not self.final_answer or not self.final_answer.strip():
                raise ValueError("answer action requires final_answer")
        elif self.final_answer is not None:
            raise ValueError("only answer action may include final_answer")
        return self


class ToolLoopStep(BaseModel):
    step: int = Field(ge=1)
    action: ToolLoopAction
    status: Literal["succeeded", "failed"]
    observation: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class ToolLoopResult(BaseModel):
    case_id: str
    completed: bool
    final_answer: str | None = None
    answer_exact: bool = False
    stop_reason: str
    steps: list[ToolLoopStep]
    discovered_evidence_ids: list[str]
    read_evidence_ids: list[str]
    cited_evidence_ids: list[str]
    unknown_evidence_id_attempts: int = 0
    invalid_action_attempts: int = 0
    insufficient_evidence_attempts: int = 0
    recovered_protocol_errors: int = 0
    final_protocol_failure: bool = False
    supporting_evidence_recall: float = Field(ge=0, le=1)


class AdaptiveEvidenceAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_target_evidence: int = Field(ge=1)
    evaluate_recommended: bool
    prioritize_answer: bool
    message: str


class QuestionComplexityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complexity: Literal[
        "simple_fact",
        "multi_hop",
        "comparison",
        "ambiguous",
    ]
    rationale_summary: str = Field(min_length=3, max_length=300)


class EvidenceSufficiencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "sufficient",
        "missing_link",
        "conflicting",
        "uncertain",
    ]
    covered_information: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rationale_summary: str = Field(min_length=3, max_length=500)


class ToolLoopPolicy(Protocol):
    def next_action(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> ToolLoopAction: ...


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def adaptive_evidence_advice(
    *,
    read_evidence_count: int,
    evaluated_evidence_count: int,
    remaining_steps: int,
    complexity: Literal[
        "simple_fact",
        "multi_hop",
        "comparison",
        "ambiguous",
    ]
    | None = None,
) -> AdaptiveEvidenceAdvice:
    """Return budget-aware advice without introducing a new hard failure."""
    if remaining_steps <= 2:
        return AdaptiveEvidenceAdvice(
            soft_target_evidence=1,
            evaluate_recommended=False,
            prioritize_answer=True,
            message=(
                "Prioritize a grounded answer now. Cite one or more relevant "
                "successfully read evidence IDs; never invent an ID merely "
                "to increase evidence count."
            ),
        )
    if complexity == "simple_fact":
        return AdaptiveEvidenceAdvice(
            soft_target_evidence=1,
            evaluate_recommended=False,
            prioritize_answer=read_evidence_count >= 1,
            message=(
                "This is a simple fact request. One directly supporting "
                "source is enough; answer once that source has been read."
            ),
        )
    target = 2 if complexity in {"multi_hop", "comparison"} else 2
    if read_evidence_count < 2:
        return AdaptiveEvidenceAdvice(
            soft_target_evidence=target,
            evaluate_recommended=False,
            prioritize_answer=False,
            message=(
                "This question needs multiple reasoning facts. The preferred "
                "next action is read_page for a relevant discovered evidence "
                "ID that is not already in the successfully read list. Do "
                "not repeat a read, evaluate yet, or answer before reaching "
                "the target unless only two actions remain. Do not use an "
                "irrelevant source merely to reach the target."
            ),
        )
    if evaluated_evidence_count < 2:
        return AdaptiveEvidenceAdvice(
            soft_target_evidence=2,
            evaluate_recommended=True,
            prioritize_answer=False,
            message=(
                "Two sources have been read. Evaluate the relevant evidence "
                "when useful for resolving support or conflict, but do not "
                "delay a well-grounded answer solely to satisfy this advice."
            ),
        )
    return AdaptiveEvidenceAdvice(
        soft_target_evidence=2,
        evaluate_recommended=False,
        prioritize_answer=False,
        message=(
            "Evidence coverage is sufficient. Answer when the evidence "
            "supports the conclusion."
        ),
    )


class FixedEvidenceToolEnvironment:
    """Deterministic search/read environment over a case's frozen evidence."""

    def __init__(self, evidence: Sequence[ToolLoopEvidence]) -> None:
        self.evidence = {item.evidence_id: item for item in evidence}
        self.discovered: set[str] = set()
        self.read: set[str] = set()
        self.evaluated: set[str] = set()

    def execute(self, action: ToolLoopAction) -> dict[str, Any]:
        if action.action == "search":
            return self._search(action.arguments)
        if action.action == "read_page":
            return self._read_page(action.arguments)
        if action.action == "evaluate_evidence":
            return self._evaluate(action)
        raise ToolExecutionError(
            "action_not_executable",
            f"{action.action} is not an environment tool",
        )

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolExecutionError("invalid_arguments", "search needs query")
        top_k = min(max(int(arguments.get("top_k", 5)), 1), 10)
        query_tokens = _tokens(query)
        ranked = sorted(
            self.evidence.values(),
            key=lambda item: (
                -len(
                    query_tokens
                    & _tokens(f"{item.title} {item.content}")
                ),
                item.evidence_id,
            ),
        )[:top_k]
        self.discovered.update(item.evidence_id for item in ranked)
        return {
            "query": query,
            "results": [
                {
                    "evidence_id": item.evidence_id,
                    "title": item.title,
                    "snippet": item.content[:240],
                    "source": item.source,
                }
                for item in ranked
            ],
        }

    def _read_page(self, arguments: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(arguments.get("evidence_id", "")).strip()
        if evidence_id not in self.evidence:
            raise ToolExecutionError(
                "unknown_evidence_id",
                f"unknown evidence ID: {evidence_id or '<empty>'}",
            )
        if evidence_id not in self.discovered:
            raise ToolExecutionError(
                "undiscovered_evidence_id",
                "read_page requires an ID returned by search",
            )
        item = self.evidence[evidence_id]
        self.read.add(evidence_id)
        return {
            "evidence_id": item.evidence_id,
            "title": item.title,
            "content": item.content,
            "source": item.source,
        }

    def _evaluate(self, action: ToolLoopAction) -> dict[str, Any]:
        requested = list(
            dict.fromkeys(
                [
                    *action.evidence_ids,
                    *[
                        str(value)
                        for value in action.arguments.get(
                            "evidence_ids", []
                        )
                    ],
                ]
            )
        )
        if not requested:
            raise ToolExecutionError(
                "invalid_arguments",
                "evaluate_evidence needs evidence_ids",
            )
        unknown = [item for item in requested if item not in self.evidence]
        if unknown:
            raise ToolExecutionError(
                "unknown_evidence_id",
                f"unknown evidence IDs: {', '.join(unknown)}",
            )
        unread = [item for item in requested if item not in self.read]
        if unread:
            raise ToolExecutionError(
                "unread_evidence_id",
                f"evidence must be read first: {', '.join(unread)}",
            )
        self.evaluated.update(requested)
        return {
            "evaluated_evidence_ids": requested,
            "evidence": [
                {
                    "evidence_id": item,
                    "content": self.evidence[item].content,
                }
                for item in requested
            ],
        }


def run_tool_loop(
    case: ToolLoopEvalCase,
    policy: ToolLoopPolicy,
    *,
    max_steps: int = 8,
    min_read_evidence: int = 1,
    require_evaluate_before_answer: bool = False,
) -> ToolLoopResult:
    environment = FixedEvidenceToolEnvironment(case.evidence)
    steps: list[ToolLoopStep] = []
    final_answer: str | None = None
    cited: list[str] = []
    unknown_attempts = 0
    invalid_attempts = 0
    insufficient_attempts = 0
    stop_reason = "max_steps"
    for step_number in range(1, max_steps + 1):
        try:
            action = policy.next_action(
                question=case.question,
                history=tuple(steps),
            )
        except Exception as error:
            invalid_attempts += 1
            steps.append(
                ToolLoopStep(
                    step=step_number,
                    action=ToolLoopAction(
                        rationale_summary="Policy output was invalid.",
                        action="search",
                        arguments={"query": case.question},
                    ),
                    status="failed",
                    error_code="invalid_policy_output",
                    observation={"message": str(error)},
                )
            )
            stop_reason = "invalid_policy_output"
            break
        if action.action == "answer":
            cited = list(dict.fromkeys(action.evidence_ids))
            unknown = [
                item for item in cited if item not in environment.evidence
            ]
            unread = [
                item for item in cited if item not in environment.read
            ]
            unknown_attempts += len(unknown)
            if unknown:
                error_code = "unknown_evidence_id"
            elif not cited:
                error_code = "missing_evidence_citation"
                invalid_attempts += 1
            elif unread:
                error_code = "unread_evidence_id"
                invalid_attempts += 1
            elif len(environment.read) < min_read_evidence:
                error_code = "insufficient_evidence_coverage"
                insufficient_attempts += 1
            elif len(cited) < min_read_evidence:
                error_code = "insufficient_answer_citations"
                insufficient_attempts += 1
            elif (
                require_evaluate_before_answer
                and not set(cited).issubset(environment.evaluated)
            ):
                error_code = "evidence_not_evaluated"
                insufficient_attempts += 1
            else:
                error_code = None
            status: Literal["succeeded", "failed"] = (
                "failed" if error_code else "succeeded"
            )
            steps.append(
                ToolLoopStep(
                    step=step_number,
                    action=action,
                    status=status,
                    error_code=error_code,
                    observation=(
                        {
                            "message": (
                                "Answer citations must be IDs returned by "
                                "search and successfully read."
                            ),
                            "discovered_evidence_ids": sorted(
                                environment.discovered
                            ),
                            "read_evidence_ids": sorted(environment.read),
                            "evaluated_evidence_ids": sorted(
                                environment.evaluated
                            ),
                            "minimum_required_evidence": min_read_evidence,
                            "evaluate_required": (
                                require_evaluate_before_answer
                            ),
                        }
                        if error_code
                        else {}
                    ),
                )
            )
            if status == "succeeded":
                final_answer = action.final_answer
                stop_reason = "completed"
                break
            stop_reason = error_code or "invalid_answer"
            continue
        try:
            observation = environment.execute(action)
            steps.append(
                ToolLoopStep(
                    step=step_number,
                    action=action,
                    status="succeeded",
                    observation=observation,
                )
            )
        except ToolExecutionError as error:
            if "evidence_id" in error.code:
                unknown_attempts += 1
            else:
                invalid_attempts += 1
            steps.append(
                ToolLoopStep(
                    step=step_number,
                    action=action,
                    status="failed",
                    error_code=error.code,
                    observation={"message": str(error)},
                )
            )

    gold_ids = {
        evidence_id
        for claim in case.gold_claims
        for evidence_id in claim.supporting_evidence_ids
    }
    covered = gold_ids & (environment.read | set(cited))
    recall = len(covered) / len(gold_ids) if gold_ids else 1.0
    completed = stop_reason == "completed"
    recovered_errors = sum(
        item.status == "failed" for item in steps
    ) if completed else 0
    return ToolLoopResult(
        case_id=case.case_id,
        completed=completed,
        final_answer=final_answer,
        answer_exact=(
            completed
            and _normalize(final_answer or "")
            in {
                _normalize(case.expected_answer),
                *[_normalize(item) for item in case.answer_aliases],
            }
        ),
        stop_reason=stop_reason,
        steps=steps,
        discovered_evidence_ids=sorted(environment.discovered),
        read_evidence_ids=sorted(environment.read),
        cited_evidence_ids=cited,
        unknown_evidence_id_attempts=unknown_attempts,
        invalid_action_attempts=invalid_attempts,
        insufficient_evidence_attempts=insufficient_attempts,
        recovered_protocol_errors=recovered_errors,
        final_protocol_failure=(
            not completed
            and stop_reason
            in {
                "invalid_policy_output",
                "unknown_evidence_id",
                "missing_evidence_citation",
                "unread_evidence_id",
                "insufficient_evidence_coverage",
                "insufficient_answer_citations",
                "evidence_not_evaluated",
            }
        ),
        supporting_evidence_recall=recall,
    )


def summarize_tool_loop_results(
    results: Sequence[ToolLoopResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("tool-loop evaluation needs at least one result")
    count = len(results)
    metrics = {
        "cases": count,
        "completed": sum(item.completed for item in results),
        "completion_rate": sum(item.completed for item in results) / count,
        "answer_exact": sum(item.answer_exact for item in results),
        "answer_exact_rate": sum(item.answer_exact for item in results) / count,
        "unknown_evidence_id_attempts": sum(
            item.unknown_evidence_id_attempts for item in results
        ),
        "invalid_action_attempts": sum(
            item.invalid_action_attempts for item in results
        ),
        "insufficient_evidence_attempts": sum(
            item.insufficient_evidence_attempts for item in results
        ),
        "recovered_protocol_errors": sum(
            item.recovered_protocol_errors for item in results
        ),
        "final_protocol_failures": sum(
            item.final_protocol_failure for item in results
        ),
        "mean_steps": sum(len(item.steps) for item in results) / count,
        "mean_supporting_evidence_recall": sum(
            item.supporting_evidence_recall for item in results
        )
        / count,
    }
    checks = {
        "answer_exact_at_least_50_percent": (
            metrics["answer_exact_rate"] >= 0.50
        ),
        "completion_at_least_95_percent": metrics["completion_rate"] >= 0.95,
        "no_unknown_evidence_ids": (
            metrics["unknown_evidence_id_attempts"] == 0
        ),
        "no_invalid_actions": metrics["invalid_action_attempts"] == 0,
        "no_final_protocol_failures": (
            metrics["final_protocol_failures"] == 0
        ),
        "supporting_evidence_recall_at_least_90_percent": (
            metrics["mean_supporting_evidence_recall"] >= 0.90
        ),
    }
    return {
        "metrics": metrics,
        "quality_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_rl"
                if all(checks.values())
                else "hold_before_rl"
            ),
        },
    }


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_PATTERN.findall(
            unicodedata.normalize("NFKC", value)
        )
    }


def _normalize(value: str) -> str:
    return " ".join(sorted(_tokens(value)))
