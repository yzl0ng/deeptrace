from __future__ import annotations

import re
import statistics
import time
from collections import Counter
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.tokenizer import tokenize

CITATION_PATTERN = re.compile(r"\[E:([A-Za-z0-9._:-]+)\]")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")


class AgentMode(StrEnum):
    DIRECT = "direct"
    RAG = "rag"
    REACT = "react"
    DEEP_RESEARCH = "deep_research"


class SupportLabel(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class EvalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    title: str
    content: str
    source: str


class GoldClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)


class AgentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset: str
    split: str
    language: str
    question: str
    expected_answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    evidence: list[EvalEvidence]
    gold_claims: list[GoldClaim]
    reviewed: bool
    source_record_id: str


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    mode: AgentMode
    answer: str
    report: str
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class Claim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    claim_id: str
    evidence_id: str
    valid: bool


class SupportJudgment(BaseModel):
    claim_id: str
    matched_gold_claim_id: str | None
    label: SupportLabel
    cited_evidence_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class CaseEvaluation(BaseModel):
    case_id: str
    dataset: str
    mode: AgentMode
    answer_exact_match: float
    claims: list[Claim]
    citations: list[Citation]
    judgments: list[SupportJudgment]
    metrics: dict[str, float]
    failures: list[str] = Field(default_factory=list)
    latency_ms: float
    total_tokens: int


class ModeSummary(BaseModel):
    mode: AgentMode
    cases: int
    metrics: dict[str, float]
    average_latency_ms: float
    total_tokens: int
    failure_counts: dict[str, int]


class AgentEvaluationResult(BaseModel):
    dataset_hash: str
    split: str
    modes: list[ModeSummary]
    cases: list[CaseEvaluation]


class AgentAnswerProvider(Protocol):
    def answer(
        self,
        case: AgentEvalCase,
        mode: AgentMode,
    ) -> AgentAnswer: ...


class ClaimExtractor:
    """Deterministic extractor for reports with explicit [E:id] citations."""

    def extract(self, report: str) -> list[Claim]:
        claims: list[Claim] = []
        for sentence in SENTENCE_BOUNDARY.split(report.strip()):
            text = sentence.strip()
            if not text:
                continue
            evidence_ids = CITATION_PATTERN.findall(text)
            claim_text = CITATION_PATTERN.sub("", text).strip()
            claim_text = claim_text.strip()
            if not claim_text:
                continue
            claims.append(
                Claim(
                    claim_id=f"claim-{len(claims) + 1}",
                    text=claim_text,
                    citation_ids=list(dict.fromkeys(evidence_ids)),
                )
            )
        return claims


class CitationVerifier:
    """Gold-evidence verifier for frozen evaluation cases.

    Claim matching uses a transparent token-F1 threshold. It is a deterministic
    benchmark verifier, not a semantic NLI model.
    """

    def __init__(self, *, claim_match_threshold: float = 0.72) -> None:
        if not 0 <= claim_match_threshold <= 1:
            raise ValueError("claim_match_threshold must be between 0 and 1")
        self.claim_match_threshold = claim_match_threshold

    def verify(
        self,
        case: AgentEvalCase,
        claims: list[Claim],
    ) -> tuple[list[Citation], list[SupportJudgment]]:
        evidence_ids = {item.evidence_id for item in case.evidence}
        citations = [
            Citation(
                citation_id=f"citation-{claim.claim_id}-{index + 1}",
                claim_id=claim.claim_id,
                evidence_id=evidence_id,
                valid=evidence_id in evidence_ids,
            )
            for claim in claims
            for index, evidence_id in enumerate(claim.citation_ids)
        ]
        judgments = [
            self._judge_claim(case, claim, evidence_ids)
            for claim in claims
        ]
        return citations, judgments

    def _judge_claim(
        self,
        case: AgentEvalCase,
        claim: Claim,
        valid_evidence_ids: set[str],
    ) -> SupportJudgment:
        matched, similarity = _best_gold_claim(claim.text, case.gold_claims)
        cited = set(claim.citation_ids) & valid_evidence_ids
        if matched is None or similarity < self.claim_match_threshold:
            return SupportJudgment(
                claim_id=claim.claim_id,
                matched_gold_claim_id=None,
                label=SupportLabel.UNSUPPORTED,
                cited_evidence_ids=sorted(cited),
                reason="claim did not match a frozen gold claim",
            )

        supporting = set(matched.supporting_evidence_ids)
        contradicting = set(matched.contradicting_evidence_ids)
        if cited & contradicting:
            label = SupportLabel.CONTRADICTED
            reason = (
                "citation set contains contradicting evidence"
                if cited & supporting
                else "citation maps only to contradicting evidence"
            )
        elif not cited:
            label = SupportLabel.UNSUPPORTED
            reason = "claim has no valid citation"
        elif supporting and supporting.issubset(cited):
            label = SupportLabel.SUPPORTED
            reason = "citations cover all gold supporting evidence"
        elif cited & supporting:
            label = SupportLabel.PARTIAL
            reason = "citations cover only part of gold supporting evidence"
        else:
            label = SupportLabel.UNSUPPORTED
            reason = "valid citations do not support the matched claim"
        return SupportJudgment(
            claim_id=claim.claim_id,
            matched_gold_claim_id=matched.claim_id,
            label=label,
            cited_evidence_ids=sorted(cited),
            supporting_evidence_ids=sorted(supporting),
            reason=reason,
        )


class AgentEvalRunner:
    def __init__(
        self,
        *,
        provider: AgentAnswerProvider,
        extractor: ClaimExtractor | None = None,
        verifier: CitationVerifier | None = None,
    ) -> None:
        self.provider = provider
        self.extractor = extractor or ClaimExtractor()
        self.verifier = verifier or CitationVerifier()

    def evaluate(
        self,
        cases: list[AgentEvalCase],
        *,
        modes: list[AgentMode],
        dataset_hash: str,
        split: str = "test",
    ) -> AgentEvaluationResult:
        selected = [item for item in cases if item.split == split]
        if not selected:
            raise ValueError(f"no cases found for frozen split {split!r}")
        if any(not item.reviewed for item in selected):
            raise ValueError("formal evaluation requires reviewed cases")
        if not modes:
            raise ValueError("formal evaluation requires at least one mode")
        if len(set(modes)) != len(modes):
            raise ValueError("formal evaluation modes must be unique")
        case_results: list[CaseEvaluation] = []
        for mode in modes:
            for case in selected:
                answer = self.provider.answer(case, mode)
                if answer.case_id != case.case_id:
                    raise ValueError(
                        "answer provider returned a mismatched case_id"
                    )
                if answer.mode != mode:
                    raise ValueError(
                        "answer provider returned a mismatched mode"
                    )
                case_results.append(self._evaluate_case(case, answer))
        return AgentEvaluationResult(
            dataset_hash=dataset_hash,
            split=split,
            modes=[
                _summarize_mode(mode, case_results)
                for mode in modes
            ],
            cases=case_results,
        )

    def _evaluate_case(
        self,
        case: AgentEvalCase,
        answer: AgentAnswer,
    ) -> CaseEvaluation:
        claims = self.extractor.extract(answer.report)
        citations, judgments = self.verifier.verify(case, claims)
        metrics = _case_metrics(claims, citations, judgments)
        metrics["answer_exact_match"] = _answer_exact_match(
            answer.answer,
            [case.expected_answer, *case.answer_aliases],
        )
        failures = _failure_types(
            metrics,
            judgments,
            answer_exact_match=metrics["answer_exact_match"],
        )
        return CaseEvaluation(
            case_id=case.case_id,
            dataset=case.dataset,
            mode=answer.mode,
            answer_exact_match=metrics["answer_exact_match"],
            claims=claims,
            citations=citations,
            judgments=judgments,
            metrics=metrics,
            failures=failures,
            latency_ms=answer.latency_ms,
            total_tokens=answer.total_tokens,
        )


class ScriptedComparisonProvider:
    """Fixed mode policies used to smoke-test the evaluation pipeline."""

    def answer(
        self,
        case: AgentEvalCase,
        mode: AgentMode,
    ) -> AgentAnswer:
        started = time.perf_counter()
        gold = case.gold_claims[0]
        if mode == AgentMode.DIRECT:
            citations: list[str] = []
        elif mode == AgentMode.RAG:
            citations = gold.supporting_evidence_ids[:1]
        else:
            citations = list(gold.supporting_evidence_ids)
        citation_text = "".join(f" [E:{item}]" for item in citations)
        report = (
            f"{gold.text.rstrip('.' + chr(0x3002))}{citation_text}."
        )
        token_multiplier = {
            AgentMode.DIRECT: 1,
            AgentMode.RAG: 2,
            AgentMode.REACT: 3,
            AgentMode.DEEP_RESEARCH: 4,
        }[mode]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return AgentAnswer(
            case_id=case.case_id,
            mode=mode,
            answer=case.expected_answer,
            report=report,
            latency_ms=elapsed_ms + token_multiplier,
            prompt_tokens=20 * token_multiplier,
            completion_tokens=5 * token_multiplier,
            total_tokens=25 * token_multiplier,
        )


def _best_gold_claim(
    text: str,
    gold_claims: list[GoldClaim],
) -> tuple[GoldClaim | None, float]:
    if not gold_claims:
        return None, 0.0
    scored = [(_token_f1(text, item.text), item) for item in gold_claims]
    score, claim = max(scored, key=lambda item: item[0])
    return claim, score


def _token_f1(left: str, right: str) -> float:
    left_tokens = Counter(tokenize(_normalize(left)))
    right_tokens = Counter(tokenize(_normalize(right)))
    if not left_tokens or not right_tokens:
        return float(_normalize(left) == _normalize(right))
    overlap = sum((left_tokens & right_tokens).values())
    precision = overlap / sum(left_tokens.values())
    recall = overlap / sum(right_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold(), flags=re.UNICODE).strip()


def _answer_exact_match(answer: str, expected: list[str]) -> float:
    normalized = _normalize(answer)
    return float(any(normalized == _normalize(item) for item in expected))


def _case_metrics(
    claims: list[Claim],
    citations: list[Citation],
    judgments: list[SupportJudgment],
) -> dict[str, float]:
    claim_count = len(claims)
    citation_count = len(citations)
    cited_claims = sum(bool(item.citation_ids) for item in claims)
    valid_citations = sum(item.valid for item in citations)
    supported = sum(
        item.label == SupportLabel.SUPPORTED for item in judgments
    )
    partial = sum(item.label == SupportLabel.PARTIAL for item in judgments)
    contradicted = sum(
        item.label == SupportLabel.CONTRADICTED for item in judgments
    )
    correctly_mapped_citations = 0
    cited_supporting_evidence = 0
    total_supporting_evidence = 0
    judgment_by_claim = {item.claim_id: item for item in judgments}
    for citation in citations:
        judgment = judgment_by_claim[citation.claim_id]
        if (
            citation.valid
            and citation.evidence_id in judgment.supporting_evidence_ids
        ):
            correctly_mapped_citations += 1
    for judgment in judgments:
        supporting = set(judgment.supporting_evidence_ids)
        total_supporting_evidence += len(supporting)
        cited_supporting_evidence += len(
            supporting & set(judgment.cited_evidence_ids)
        )
    return {
        "citation_presence_rate": _ratio(cited_claims, claim_count),
        "claim_support_rate": _ratio(supported, claim_count),
        "partial_support_rate": _ratio(partial, claim_count),
        "contradiction_rate": _ratio(contradicted, claim_count),
        "unsupported_claim_rate": _ratio(
            claim_count - supported - partial - contradicted,
            claim_count,
        ),
        "citation_precision": _ratio(
            correctly_mapped_citations, citation_count
        ),
        "citation_coverage": _ratio(
            cited_supporting_evidence,
            total_supporting_evidence,
        ),
        "invalid_citation_rate": _ratio(
            citation_count - valid_citations, citation_count
        ),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _failure_types(
    metrics: dict[str, float],
    judgments: list[SupportJudgment],
    *,
    answer_exact_match: float,
) -> list[str]:
    failures: list[str] = []
    if answer_exact_match == 0:
        failures.append("answer_incorrect")
    if not judgments:
        failures.append("empty_report")
    if metrics["citation_presence_rate"] < 1:
        failures.append("missing_citation")
    if metrics["invalid_citation_rate"] > 0:
        failures.append("invalid_citation")
    if any(
        item.label == SupportLabel.PARTIAL for item in judgments
    ):
        failures.append("partial_support")
    if any(
        item.label == SupportLabel.CONTRADICTED for item in judgments
    ):
        failures.append("contradicted_claim")
    if any(
        item.label == SupportLabel.UNSUPPORTED for item in judgments
    ):
        failures.append("unsupported_claim")
    return failures


def _summarize_mode(
    mode: AgentMode,
    rows: list[CaseEvaluation],
) -> ModeSummary:
    selected = [item for item in rows if item.mode == mode]
    metric_names = list(selected[0].metrics)
    failures = Counter(
        failure for item in selected for failure in item.failures
    )
    return ModeSummary(
        mode=mode,
        cases=len(selected),
        metrics={
            name: statistics.fmean(
                item.metrics[name] for item in selected
            )
            for name in metric_names
        },
        average_latency_ms=statistics.fmean(
            item.latency_ms for item in selected
        ),
        total_tokens=sum(item.total_tokens for item in selected),
        failure_counts=dict(sorted(failures.items())),
    )
