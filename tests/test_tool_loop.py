from __future__ import annotations

from collections.abc import Sequence

from app.agentic.tool_loop import (
    EvidenceSufficiencyAssessment,
    ToolLoopAction,
    ToolLoopEvalCase,
    ToolLoopEvidence,
    ToolLoopGoldClaim,
    ToolLoopStep,
    adaptive_evidence_advice,
    run_tool_loop,
    summarize_tool_loop_results,
)
def _case() -> ToolLoopEvalCase:
    return ToolLoopEvalCase(
        case_id="case-1",
        dataset="fixed",
        split="dev",
        language="en",
        question="What combines ranked lists?",
        expected_answer="RRF",
        evidence=[
            ToolLoopEvidence(
                evidence_id="e1",
                title="RRF",
                content="RRF combines ranked result lists.",
                source="fixture",
            )
        ],
        gold_claims=[
            ToolLoopGoldClaim(
                claim_id="g1",
                text="The answer is RRF.",
                supporting_evidence_ids=["e1"],
            )
        ],
        reviewed=True,
        source_record_id="fixture",
    )


class ScriptedPolicy:
    def next_action(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> ToolLoopAction:
        if not history:
            return ToolLoopAction(
                rationale_summary="Search the frozen evidence.",
                action="search",
                arguments={"query": question},
            )
        if len(history) == 1:
            return ToolLoopAction(
                rationale_summary="Read the discovered evidence.",
                action="read_page",
                arguments={"evidence_id": "e1"},
                evidence_ids=["e1"],
            )
        if len(history) == 2:
            return ToolLoopAction(
                rationale_summary="Evaluate the read evidence.",
                action="evaluate_evidence",
                arguments={"evidence_ids": ["e1"]},
                evidence_ids=["e1"],
            )
        return ToolLoopAction(
            rationale_summary="Return the grounded answer.",
            action="answer",
            evidence_ids=["e1"],
            final_answer="RRF",
        )


def test_real_tool_loop_executes_actions_and_scores_grounding() -> None:
    result = run_tool_loop(_case(), ScriptedPolicy())

    assert result.completed is True
    assert result.answer_exact is True
    assert result.read_evidence_ids == ["e1"]
    assert result.supporting_evidence_recall == 1
    summary = summarize_tool_loop_results([result])
    assert summary["quality_gate"]["passed"] is True


class UnknownEvidencePolicy:
    def next_action(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> ToolLoopAction:
        return ToolLoopAction(
            rationale_summary="Read an invented identifier.",
            action="read_page",
            arguments={"evidence_id": "invented"},
        )


def test_tool_loop_rejects_unknown_evidence_ids() -> None:
    result = run_tool_loop(
        _case(),
        UnknownEvidencePolicy(),
        max_steps=1,
    )

    assert result.completed is False
    assert result.unknown_evidence_id_attempts == 1
    assert result.steps[0].error_code == "unknown_evidence_id"


class RecoveringAnswerPolicy(ScriptedPolicy):
    def next_action(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> ToolLoopAction:
        if len(history) == 1:
            return ToolLoopAction(
                rationale_summary="Premature answer with unread evidence.",
                action="answer",
                evidence_ids=["e1"],
                final_answer="RRF",
            )
        if len(history) == 2 and history[-1].status == "failed":
            return ToolLoopAction(
                rationale_summary="Read evidence after controller feedback.",
                action="read_page",
                arguments={"evidence_id": "e1"},
                evidence_ids=["e1"],
            )
        if len(history) == 3:
            return ToolLoopAction(
                rationale_summary="Return a corrected grounded answer.",
                action="answer",
                evidence_ids=["e1"],
                final_answer="RRF",
            )
        return super().next_action(question=question, history=history)


def test_invalid_answer_is_blocked_and_policy_can_recover() -> None:
    result = run_tool_loop(_case(), RecoveringAnswerPolicy())

    assert result.completed is True
    assert result.answer_exact is True
    assert result.recovered_protocol_errors == 1
    assert result.final_protocol_failure is False


def test_evidence_gate_requires_two_read_and_evaluated_items() -> None:
    case = _case().model_copy(
        update={
            "evidence": [
                *_case().evidence,
                ToolLoopEvidence(
                    evidence_id="e2",
                    title="Fusion",
                    content="RRF is a rank-fusion method.",
                    source="fixture",
                ),
            ],
            "gold_claims": [
                ToolLoopGoldClaim(
                    claim_id="g1",
                    text="The answer is RRF.",
                    supporting_evidence_ids=["e1", "e2"],
                )
            ],
        }
    )

    class CoveragePolicy:
        def next_action(
            self,
            *,
            question: str,
            history: Sequence[ToolLoopStep],
        ) -> ToolLoopAction:
            actions = [
                ToolLoopAction(
                    rationale_summary="Search for both evidence items.",
                    action="search",
                    arguments={"query": question},
                ),
                ToolLoopAction(
                    rationale_summary="Read the first evidence item.",
                    action="read_page",
                    arguments={"evidence_id": "e1"},
                    evidence_ids=["e1"],
                ),
                ToolLoopAction(
                    rationale_summary="Read the second evidence item.",
                    action="read_page",
                    arguments={"evidence_id": "e2"},
                    evidence_ids=["e2"],
                ),
                ToolLoopAction(
                    rationale_summary="Evaluate both evidence items.",
                    action="evaluate_evidence",
                    arguments={"evidence_ids": ["e1", "e2"]},
                    evidence_ids=["e1", "e2"],
                ),
                ToolLoopAction(
                    rationale_summary="Answer from both evidence items.",
                    action="answer",
                    evidence_ids=["e1", "e2"],
                    final_answer="RRF",
                ),
            ]
            return actions[len(history)]

    result = run_tool_loop(
        case,
        CoveragePolicy(),
        min_read_evidence=2,
        require_evaluate_before_answer=True,
    )

    assert result.completed is True
    assert result.supporting_evidence_recall == 1


def test_adaptive_evidence_advice_targets_coverage_without_hard_block() -> None:
    early = adaptive_evidence_advice(
        read_evidence_count=1,
        evaluated_evidence_count=0,
        remaining_steps=5,
    )
    assert early.soft_target_evidence == 2
    assert early.evaluate_recommended is False
    assert early.prioritize_answer is False

    covered = adaptive_evidence_advice(
        read_evidence_count=2,
        evaluated_evidence_count=0,
        remaining_steps=4,
    )
    assert covered.evaluate_recommended is True

    budget_limited = adaptive_evidence_advice(
        read_evidence_count=1,
        evaluated_evidence_count=0,
        remaining_steps=2,
    )
    assert budget_limited.soft_target_evidence == 1
    assert budget_limited.evaluate_recommended is False
    assert budget_limited.prioritize_answer is True


def test_complexity_aware_advice_distinguishes_simple_and_multi_hop() -> None:
    simple = adaptive_evidence_advice(
        read_evidence_count=1,
        evaluated_evidence_count=0,
        remaining_steps=5,
        complexity="simple_fact",
    )
    assert simple.soft_target_evidence == 1
    assert simple.prioritize_answer is True

    multi_hop = adaptive_evidence_advice(
        read_evidence_count=1,
        evaluated_evidence_count=0,
        remaining_steps=5,
        complexity="multi_hop",
    )
    assert multi_hop.soft_target_evidence == 2
    assert multi_hop.prioritize_answer is False


def test_evidence_sufficiency_assessment_rejects_extra_fields() -> None:
    assessment = EvidenceSufficiencyAssessment(
        status="missing_link",
        covered_information=["The person's birth date is supported."],
        missing_information=[
            "Evidence connecting the person to the film is missing."
        ],
        rationale_summary="Only the final attribute is covered.",
    )
    assert assessment.status == "missing_link"
