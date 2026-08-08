from __future__ import annotations

import pytest

from app.evaluation.agentic import (
    AgentAnswer,
    AgentEvalCase,
    AgentEvalRunner,
    AgentMode,
    CitationVerifier,
    Claim,
    ClaimExtractor,
    EvalEvidence,
    GoldClaim,
    ScriptedComparisonProvider,
    SupportLabel,
)


def _case(*, reviewed: bool = True) -> AgentEvalCase:
    return AgentEvalCase(
        case_id="case-1",
        dataset="fixed",
        split="test",
        language="en",
        question="What combines lexical and semantic ranks?",
        expected_answer="RRF",
        evidence=[
            EvalEvidence(
                evidence_id="ev-1",
                title="Lexical",
                content="BM25 provides lexical ranking.",
                source="fixture",
            ),
            EvalEvidence(
                evidence_id="ev-2",
                title="Fusion",
                content="RRF combines ranked result lists.",
                source="fixture",
            ),
            EvalEvidence(
                evidence_id="ev-3",
                title="Contradiction",
                content="A distractor says ranks should not be fused.",
                source="fixture",
            ),
        ],
        gold_claims=[
            GoldClaim(
                claim_id="gold-1",
                text="The answer to the question is RRF.",
                supporting_evidence_ids=["ev-1", "ev-2"],
                contradicting_evidence_ids=["ev-3"],
            )
        ],
        reviewed=reviewed,
        source_record_id="fixture-1",
    )


def test_claim_extractor_attaches_explicit_evidence_ids() -> None:
    claims = ClaimExtractor().extract(
        "RRF combines ranks [E:ev-1][E:ev-2]. "
        "It avoids raw score addition [E:ev-2]."
    )
    assert len(claims) == 2
    assert claims[0].citation_ids == ["ev-1", "ev-2"]
    assert "[E:" not in claims[0].text


def test_claim_extractor_splits_chinese_sentence_boundaries() -> None:
    claims = ClaimExtractor().extract(
        "\u7b2c\u4e00\u6761\u4e8b\u5b9e [E:ev-1]\u3002 "
        "\u7b2c\u4e8c\u6761\u4e8b\u5b9e [E:ev-2]\uff01"
    )

    assert [claim.citation_ids for claim in claims] == [
        ["ev-1"],
        ["ev-2"],
    ]


def test_verifier_separates_support_labels_and_invalid_citations() -> None:
    case = _case()
    verifier = CitationVerifier(claim_match_threshold=0.5)

    citations, judgments = verifier.verify(
        case,
        [
            Claim(
                claim_id="claim-0",
                text="The answer to the question is RRF.",
                citation_ids=["ev-1", "ev-2"],
            ),
            Claim(
                claim_id="claim-1",
                text="The answer to the question is RRF.",
                citation_ids=["ev-1"],
            ),
            Claim(
                claim_id="claim-2",
                text="The answer to the question is RRF.",
                citation_ids=["ev-3"],
            ),
            Claim(
                claim_id="claim-3",
                text="An unrelated unsupported statement.",
                citation_ids=["missing"],
            ),
        ],
    )

    assert judgments[0].label == SupportLabel.SUPPORTED
    assert judgments[1].label == SupportLabel.PARTIAL
    assert judgments[2].label == SupportLabel.CONTRADICTED
    assert judgments[3].label == SupportLabel.UNSUPPORTED
    assert citations[-1].valid is False


def test_contradicting_citation_takes_precedence_over_support() -> None:
    _, judgments = CitationVerifier(claim_match_threshold=0.5).verify(
        _case(),
        [
            Claim(
                claim_id="claim-mixed",
                text="The answer to the question is RRF.",
                citation_ids=["ev-1", "ev-2", "ev-3"],
            )
        ],
    )

    assert judgments[0].label == SupportLabel.CONTRADICTED
    assert "contains contradicting evidence" in judgments[0].reason


def test_runner_reports_citation_presence_separately_from_support() -> None:
    result = AgentEvalRunner(
        provider=ScriptedComparisonProvider()
    ).evaluate(
        [_case()],
        modes=[
            AgentMode.DIRECT,
            AgentMode.RAG,
            AgentMode.REACT,
            AgentMode.DEEP_RESEARCH,
        ],
        dataset_hash="fixed-hash",
    )
    summaries = {item.mode: item for item in result.modes}
    assert (
        summaries[AgentMode.DIRECT].metrics[
            "citation_presence_rate"
        ]
        == 0
    )
    assert (
        summaries[AgentMode.DIRECT].metrics["answer_exact_match"]
        == 1
    )
    assert summaries[AgentMode.RAG].metrics["partial_support_rate"] == 1
    assert summaries[AgentMode.RAG].metrics["citation_coverage"] == 0.5
    assert (
        summaries[AgentMode.REACT].metrics["claim_support_rate"]
        == 1
    )
    assert (
        summaries[AgentMode.DEEP_RESEARCH].metrics[
            "claim_support_rate"
        ]
        == 1
    )


def test_formal_runner_rejects_unreviewed_or_non_test_data() -> None:
    runner = AgentEvalRunner(provider=ScriptedComparisonProvider())
    with pytest.raises(ValueError, match="reviewed"):
        runner.evaluate(
            [_case(reviewed=False)],
            modes=[AgentMode.DIRECT],
            dataset_hash="hash",
        )
    with pytest.raises(ValueError, match="no cases"):
        runner.evaluate(
            [_case()],
            modes=[AgentMode.DIRECT],
            dataset_hash="hash",
            split="dev",
        )


def test_formal_runner_rejects_empty_duplicate_or_misattributed_modes() -> None:
    runner = AgentEvalRunner(provider=ScriptedComparisonProvider())
    with pytest.raises(ValueError, match="at least one mode"):
        runner.evaluate([_case()], modes=[], dataset_hash="hash")
    with pytest.raises(ValueError, match="unique"):
        runner.evaluate(
            [_case()],
            modes=[AgentMode.DIRECT, AgentMode.DIRECT],
            dataset_hash="hash",
        )

    class MismatchedProvider:
        def answer(
            self,
            case: AgentEvalCase,
            mode: AgentMode,
        ) -> AgentAnswer:
            return AgentAnswer(
                case_id="wrong-case",
                mode=AgentMode.RAG,
                answer=case.expected_answer,
                report="The answer to the question is RRF.",
                latency_ms=0,
            )

    with pytest.raises(ValueError, match="case_id"):
        AgentEvalRunner(provider=MismatchedProvider()).evaluate(
            [_case()],
            modes=[AgentMode.DIRECT],
            dataset_hash="hash",
        )

    class MismatchedModeProvider(MismatchedProvider):
        def answer(
            self,
            case: AgentEvalCase,
            mode: AgentMode,
        ) -> AgentAnswer:
            answer = super().answer(case, mode)
            return answer.model_copy(update={"case_id": case.case_id})

    with pytest.raises(ValueError, match="mode"):
        AgentEvalRunner(provider=MismatchedModeProvider()).evaluate(
            [_case()],
            modes=[AgentMode.DIRECT],
            dataset_hash="hash",
        )


def test_empty_report_is_an_explicit_failure() -> None:
    class EmptyReportProvider:
        def answer(
            self,
            case: AgentEvalCase,
            mode: AgentMode,
        ) -> AgentAnswer:
            return AgentAnswer(
                case_id=case.case_id,
                mode=mode,
                answer=case.expected_answer,
                report="",
                latency_ms=0,
            )

    result = AgentEvalRunner(provider=EmptyReportProvider()).evaluate(
        [_case()],
        modes=[AgentMode.DIRECT],
        dataset_hash="hash",
    )

    assert "empty_report" in result.cases[0].failures
    assert "missing_citation" in result.cases[0].failures
