from __future__ import annotations

from collections import Counter
from copy import deepcopy
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from app.evaluation.retrieval_metrics import (
    ndcg_at_k,
    reciprocal_rank,
)


class FailureType(StrEnum):
    CORPUS_MISSING = "corpus_missing"
    PARSING_FAILURE = "parsing_failure"
    SCANNED_PDF_NO_TEXT = "scanned_pdf_no_text"
    ENCODING_FAILURE = "encoding_failure"
    METADATA_MISSING = "metadata_missing"
    CHUNK_TOO_SHORT = "chunk_too_short"
    CHUNK_TOO_LONG = "chunk_too_long"
    BOUNDARY_INFORMATION_LOSS = "boundary_information_loss"
    DUPLICATE_CHUNK = "duplicate_chunk"
    CONTEXT_FRAGMENTATION = "context_fragmentation"
    LEXICAL_MISMATCH = "lexical_mismatch"
    TOKENIZATION_FAILURE = "tokenization_failure"
    ACRONYM_FAILURE = "acronym_failure"
    EXACT_TERM_FALSE_POSITIVE = "exact_term_false_positive"
    DENSE_FALSE_POSITIVE = "dense_false_positive"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    EMBEDDING_DOMAIN_MISMATCH = "embedding_domain_mismatch"
    FUSION_IMPROVEMENT = "fusion_improvement"
    FUSION_REGRESSION = "fusion_regression"
    CANDIDATE_K_TOO_SMALL = "candidate_k_too_small"
    WEAK_PATH_INTERFERENCE = "weak_path_interference"
    RERANKER_IMPROVEMENT = "reranker_improvement"
    RERANKER_REGRESSION = "reranker_regression"
    HARD_NEGATIVE_CONFUSION = "hard_negative_confusion"
    CANDIDATE_MISSING = "candidate_missing_before_rerank"
    CONTEXT_SELECTION_FAILURE = "context_selection_failure"
    CONTEXT_NOISE = "context_noise"
    EVIDENCE_TRUNCATED = "evidence_truncated"
    REDUNDANT_CONTEXT = "redundant_context"
    GENERATION_HALLUCINATION = "generation_hallucination"
    INSTRUCTION_FOLLOWING_FAILURE = "instruction_following_failure"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    ANSWER_INCOMPLETE = "answer_incomplete"
    CITATION_INVALID_ID = "citation_invalid_id"
    CITATION_WRONG_EVIDENCE = "citation_wrong_evidence"
    CITATION_INCOMPLETE = "citation_incomplete"
    CITATION_MISALIGNED = "citation_misaligned"
    CITATION_MISSING = "citation_missing"
    ABSTENTION_FAILURE = "abstention_failure"
    FALSE_ABSTENTION = "false_abstention"
    NO_ANSWER_FALSE_POSITIVE = "no_answer_false_positive"
    CORRECT_ABSTENTION = "correct_abstention"
    PROMPT_INJECTION_RISK = "prompt_injection_risk"
    ANNOTATION_UNCERTAIN = "annotation_uncertain"


PIPELINE_STAGE = {
    FailureType.CORPUS_MISSING: "corpus",
    FailureType.PARSING_FAILURE: "parsing",
    FailureType.SCANNED_PDF_NO_TEXT: "parsing",
    FailureType.ENCODING_FAILURE: "parsing",
    FailureType.METADATA_MISSING: "parsing",
    FailureType.CHUNK_TOO_SHORT: "chunking",
    FailureType.CHUNK_TOO_LONG: "chunking",
    FailureType.BOUNDARY_INFORMATION_LOSS: "chunking",
    FailureType.DUPLICATE_CHUNK: "chunking",
    FailureType.CONTEXT_FRAGMENTATION: "chunking",
    FailureType.LEXICAL_MISMATCH: "bm25",
    FailureType.TOKENIZATION_FAILURE: "bm25",
    FailureType.ACRONYM_FAILURE: "bm25",
    FailureType.EXACT_TERM_FALSE_POSITIVE: "bm25",
    FailureType.DENSE_FALSE_POSITIVE: "dense",
    FailureType.SEMANTIC_MISMATCH: "dense",
    FailureType.EMBEDDING_DOMAIN_MISMATCH: "dense",
    FailureType.FUSION_IMPROVEMENT: "rrf",
    FailureType.FUSION_REGRESSION: "rrf",
    FailureType.CANDIDATE_K_TOO_SMALL: "rrf",
    FailureType.WEAK_PATH_INTERFERENCE: "rrf",
    FailureType.RERANKER_IMPROVEMENT: "reranker",
    FailureType.RERANKER_REGRESSION: "reranker",
    FailureType.HARD_NEGATIVE_CONFUSION: "reranker",
    FailureType.CANDIDATE_MISSING: "retrieval",
    FailureType.CONTEXT_SELECTION_FAILURE: "context",
    FailureType.CONTEXT_NOISE: "context",
    FailureType.EVIDENCE_TRUNCATED: "context",
    FailureType.REDUNDANT_CONTEXT: "context",
    FailureType.GENERATION_HALLUCINATION: "generation",
    FailureType.INSTRUCTION_FOLLOWING_FAILURE: "generation",
    FailureType.UNSUPPORTED_CLAIM: "generation",
    FailureType.ANSWER_INCOMPLETE: "generation",
    FailureType.CITATION_INVALID_ID: "citation",
    FailureType.CITATION_WRONG_EVIDENCE: "citation",
    FailureType.CITATION_INCOMPLETE: "citation",
    FailureType.CITATION_MISALIGNED: "citation",
    FailureType.CITATION_MISSING: "citation",
    FailureType.ABSTENTION_FAILURE: "abstention",
    FailureType.FALSE_ABSTENTION: "abstention",
    FailureType.NO_ANSWER_FALSE_POSITIVE: "abstention",
    FailureType.CORRECT_ABSTENTION: "abstention",
    FailureType.PROMPT_INJECTION_RISK: "security",
    FailureType.ANNOTATION_UNCERTAIN: "annotation",
}
STAGE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "corpus",
            "parsing",
            "chunking",
            "bm25",
            "dense",
            "rrf",
            "retrieval",
            "reranker",
            "context",
            "generation",
            "citation",
            "abstention",
            "security",
            "annotation",
        )
    )
}
NON_FAILURE_SIGNALS = {
    FailureType.FUSION_IMPROVEMENT,
    FailureType.RERANKER_IMPROVEMENT,
    FailureType.CORRECT_ABSTENTION,
}


FIXES = {
    FailureType.LEXICAL_MISMATCH: (
        "保留 Dense/RRF，或对缩写和同义表达做可审计的查询改写。",
        "查询改写会增加延迟，也可能改变原始意图。",
    ),
    FailureType.DENSE_FALSE_POSITIVE: (
        "保留 BM25 精确词召回，并增加领域困难负例评测。",
        "领域数据与训练成本会上升。",
    ),
    FailureType.FUSION_REGRESSION: (
        "在固定验证集上分析 rank_constant 与候选深度，不覆盖现有基线。",
        "调参可能只适合当前数据分布，并增加实验成本。",
    ),
    FailureType.CANDIDATE_MISSING: (
        "扩大召回 candidate_k 或改善上游检索器。",
        "候选增大会增加 Dense、融合和重排延迟。",
    ),
    FailureType.RERANKER_IMPROVEMENT: (
        "保留该困难样例作为回归测试。",
        "Cross-Encoder 会增加计算与端到端延迟。",
    ),
    FailureType.RERANKER_REGRESSION: (
        "补充困难负例并比较不同候选深度或领域 Reranker。",
        "训练、标注和在线推理成本都会增加。",
    ),
    FailureType.CITATION_INVALID_ID: (
        "继续在服务端拦截不在 Context 中的引用 ID。",
        "只能保证 ID 合法，不能自动证明引用内容正确。",
    ),
    FailureType.CITATION_MISSING: (
        "强化逐结论引用约束并增加引用完整性人工评测。",
        "回答可能更保守，且人工标注成本较高。",
    ),
    FailureType.NO_ANSWER_FALSE_POSITIVE: (
        "建立无答案集并单独校准拒答策略。",
        "更强拒答会提高可回答问题上的误拒率。",
    ),
    FailureType.FALSE_ABSTENTION: (
        "检查 Top-5 证据完整性并校准拒答提示与阈值。",
        "减少误拒可能同时增加无答案问题的误答。",
    ),
    FailureType.CORRECT_ABSTENTION: (
        "保留为拒答回归案例。",
        "它只证明当前案例行为正确，不代表总体拒答准确率。",
    ),
    FailureType.ANNOTATION_UNCERTAIN: (
        "补充人工相关性、参考答案与证据标注后再确认根因。",
        "需要领域专家时间，不能自动化替代。",
    ),
}


class FailureAnalyzer:
    """Generate deterministic review candidates, never human judgments."""

    def analyze_query(self, case: Mapping[str, Any]) -> dict[str, Any]:
        candidate = deepcopy(dict(case))
        labels: list[FailureType] = []
        evidence: list[str] = list(candidate.get("evidence", []))
        expected = candidate.get("expected") or {}
        retrieval = candidate.get("retrieval") or {}
        relevance = expected.get("relevance") or {
            item_id: 1
            for item_id in expected.get("relevant_chunk_ids", [])
        }

        if relevance:
            self._classify_retrieval(retrieval, relevance, labels, evidence)
            self._classify_reranker(retrieval, relevance, labels, evidence)
        elif expected.get("answerable") is not False:
            self._add(
                labels,
                evidence,
                FailureType.ANNOTATION_UNCERTAIN,
                "No human qrels or expected source IDs are available.",
            )

        self._classify_rag(candidate.get("rag") or {}, expected, labels, evidence)
        for signal in candidate.get("observed_signals", []):
            try:
                label = FailureType(signal)
            except ValueError:
                continue
            self._add(
                labels,
                evidence,
                label,
                f"Observed pipeline signal: {label.value}.",
            )

        candidate["failure_types"] = [item.value for item in labels]
        candidate["pipeline_stages"] = sorted(
            {PIPELINE_STAGE[item] for item in labels},
            key=STAGE_ORDER.__getitem__,
        )
        candidate["evidence"] = evidence
        candidate["reviewed"] = bool(candidate.get("reviewed", False))
        candidate.setdefault("review_notes", "")
        if labels:
            first = next(
                (item for item in labels if item not in NON_FAILURE_SIGNALS),
                labels[0],
            )
            candidate.setdefault(
                "root_cause",
                (
                    f"自动候选：最早可观察信号位于 "
                    f"{PIPELINE_STAGE[first]} 层；尚未等同于人工确认根因。"
                ),
            )
            fix, tradeoff = FIXES.get(
                first,
                (
                    "针对该层补充可复现测试与人工证据。",
                    "需要额外标注、计算或系统复杂度。",
                ),
            )
            candidate.setdefault("possible_fix", [fix])
            candidate.setdefault("tradeoff", tradeoff)
        return candidate

    def analyze_many(
        self, cases: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for case in cases:
            candidate = self.analyze_query(case)
            case_id = str(
                candidate.get("case_id")
                or candidate.get("query_id")
                or ""
            )
            if not case_id:
                raise ValueError("Every failure-analysis case needs an ID.")
            by_id.setdefault(case_id, candidate)
        return [by_id[key] for key in sorted(by_id)]

    def compare_retrievers(
        self,
        retrieval: Mapping[str, Any],
        relevance: Mapping[str, int],
    ) -> dict[str, float]:
        return {
            name: ndcg_at_k(
                details.get("ranked_chunk_ids", []), relevance, 5
            )
            for name, details in retrieval.items()
            if name in {"bm25", "dense", "dense_exact", "rrf"}
        }

    def compare_rag_variants(
        self, rag: Mapping[str, Any]
    ) -> dict[str, Any]:
        baseline = rag.get("baseline") or {}
        reranked = rag.get("reranked") or {}
        return {
            "answer_changed": baseline.get("answer") != reranked.get("answer"),
            "citations_changed": baseline.get("citations")
            != reranked.get("citations"),
            "abstention_changed": baseline.get("abstained")
            != reranked.get("abstained"),
            "context_changed": baseline.get("context_chunk_ids")
            != reranked.get("context_chunk_ids"),
            "baseline_total_ms": _nested(baseline, "latency", "total_ms"),
            "reranked_total_ms": _nested(reranked, "latency", "total_ms"),
            "baseline_tokens": _nested(baseline, "usage", "total_tokens"),
            "reranked_tokens": _nested(reranked, "usage", "total_tokens"),
            "cost": None,
        }

    def classify_candidates(
        self, cases: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.analyze_many(cases)

    def summarize(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        experiment_id: str,
        generated_at: str,
        corpus_version: str,
        query_set_version: str,
        baseline_rag: Mapping[str, Any] | None = None,
        reranked_rag: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        failure_counts: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter()
        for candidate in candidates:
            failure_counts.update(candidate.get("failure_types", []))
            stage_counts.update(candidate.get("pipeline_stages", []))
        observed = set(failure_counts)
        tracked = [
            FailureType.FUSION_IMPROVEMENT,
            FailureType.FUSION_REGRESSION,
            FailureType.RERANKER_IMPROVEMENT,
            FailureType.RERANKER_REGRESSION,
            FailureType.CANDIDATE_MISSING,
            FailureType.CITATION_INVALID_ID,
            FailureType.CITATION_MISSING,
            FailureType.NO_ANSWER_FALSE_POSITIVE,
            FailureType.FALSE_ABSTENTION,
            FailureType.CORRECT_ABSTENTION,
            FailureType.PROMPT_INJECTION_RISK,
        ]
        reviewed = sum(bool(item.get("reviewed")) for item in candidates)
        return {
            "experiment_id": experiment_id,
            "generated_at": generated_at,
            "corpus_version": corpus_version,
            "query_set_version": query_set_version,
            "queries_analyzed": len(candidates),
            "reviewed_cases": reviewed,
            "unreviewed_candidates": len(candidates) - reviewed,
            "failure_counts": dict(sorted(failure_counts.items())),
            "pipeline_stage_counts": dict(sorted(stage_counts.items())),
            "baseline_rag": dict(baseline_rag or {}),
            "reranked_rag": dict(reranked_rag or {}),
            "reranker_improvements": failure_counts[
                FailureType.RERANKER_IMPROVEMENT.value
            ],
            "reranker_regressions": failure_counts[
                FailureType.RERANKER_REGRESSION.value
            ],
            "no_observed_case": [
                item.value for item in tracked if item.value not in observed
            ],
            "limitations": [
                "Automatic labels are review candidates, not proven root causes.",
                "Generation correctness, faithfulness, and citation correctness "
                "require human evidence review.",
                "SciFact retrieval judgments and SearchLab demo RAG cases have "
                "different corpora and must not be aggregated into one quality score.",
            ],
        }

    def _classify_retrieval(
        self,
        retrieval: Mapping[str, Any],
        relevance: Mapping[str, int],
        labels: list[FailureType],
        evidence: list[str],
    ) -> None:
        bm25 = retrieval.get("bm25") or {}
        dense = retrieval.get("dense") or retrieval.get("dense_exact") or {}
        rrf = retrieval.get("rrf") or {}
        bm_ids = bm25.get("ranked_chunk_ids", [])
        dense_ids = dense.get("ranked_chunk_ids", [])
        rrf_ids = rrf.get("ranked_chunk_ids", [])
        bm_recall = _recall(bm_ids, relevance, 5)
        dense_recall = _recall(dense_ids, relevance, 5)
        if dense_recall > bm_recall:
            self._add(
                labels,
                evidence,
                FailureType.LEXICAL_MISMATCH,
                f"Dense Recall@5={dense_recall:.3f} > "
                f"BM25 Recall@5={bm_recall:.3f}.",
            )
        if bm_recall > 0 and dense_recall == 0:
            self._add(
                labels,
                evidence,
                FailureType.DENSE_FALSE_POSITIVE,
                "BM25 Top-5 contains judged evidence while Dense Top-5 does not.",
            )
        if rrf_ids:
            bm_ndcg = ndcg_at_k(bm_ids, relevance, 5)
            dense_ndcg = ndcg_at_k(dense_ids, relevance, 5)
            rrf_ndcg = ndcg_at_k(rrf_ids, relevance, 5)
            best = max(bm_ndcg, dense_ndcg)
            if rrf_ndcg < best:
                self._add(
                    labels,
                    evidence,
                    FailureType.FUSION_REGRESSION,
                    f"RRF nDCG@5={rrf_ndcg:.3f} < best single path "
                    f"nDCG@5={best:.3f}.",
                )
            elif rrf_ndcg > best:
                self._add(
                    labels,
                    evidence,
                    FailureType.FUSION_IMPROVEMENT,
                    f"RRF nDCG@5={rrf_ndcg:.3f} > best single path "
                    f"nDCG@5={best:.3f}.",
                )

    def _classify_reranker(
        self,
        retrieval: Mapping[str, Any],
        relevance: Mapping[str, int],
        labels: list[FailureType],
        evidence: list[str],
    ) -> None:
        reranker = retrieval.get("reranker") or {}
        before = reranker.get("before", [])
        after = reranker.get("after", [])
        if not before and not after:
            return
        relevant = {key for key, grade in relevance.items() if grade > 0}
        if relevant.isdisjoint(before):
            self._add(
                labels,
                evidence,
                FailureType.CANDIDATE_MISSING,
                "No judged relevant ID is present before reranking.",
            )
            return
        before_ndcg = ndcg_at_k(before, relevance, 5)
        after_ndcg = ndcg_at_k(after, relevance, 5)
        before_mrr = reciprocal_rank(before, relevance)
        after_mrr = reciprocal_rank(after, relevance)
        if after_ndcg > before_ndcg or after_mrr > before_mrr:
            self._add(
                labels,
                evidence,
                FailureType.RERANKER_IMPROVEMENT,
                f"Reranker changed nDCG@5 {before_ndcg:.3f}→"
                f"{after_ndcg:.3f}, MRR {before_mrr:.3f}→{after_mrr:.3f}.",
            )
        elif after_ndcg < before_ndcg or after_mrr < before_mrr:
            self._add(
                labels,
                evidence,
                FailureType.RERANKER_REGRESSION,
                f"Reranker changed nDCG@5 {before_ndcg:.3f}→"
                f"{after_ndcg:.3f}, MRR {before_mrr:.3f}→{after_mrr:.3f}.",
            )

    def _classify_rag(
        self,
        rag: Mapping[str, Any],
        expected: Mapping[str, Any],
        labels: list[FailureType],
        evidence: list[str],
    ) -> None:
        variants = [
            (name, rag.get(name) or {})
            for name in ("baseline", "reranked")
            if rag.get(name)
        ]
        answerable = expected.get("answerable")
        for name, variant in variants:
            context_ids = set(variant.get("context_chunk_ids", []))
            invalid = list(variant.get("invalid_citation_ids", []))
            citations = list(variant.get("citations", []))
            abstained = bool(variant.get("abstained"))
            if invalid or any(item not in context_ids for item in citations):
                self._add(
                    labels,
                    evidence,
                    FailureType.CITATION_INVALID_ID,
                    f"{name} has citation IDs outside its Context Top-5: "
                    f"{invalid or sorted(set(citations) - context_ids)}.",
                )
            if answerable is True and not abstained and not citations:
                self._add(
                    labels,
                    evidence,
                    FailureType.CITATION_MISSING,
                    f"{name} answered an answerable query without a citation.",
                )
            if answerable is False and not abstained:
                self._add(
                    labels,
                    evidence,
                    FailureType.NO_ANSWER_FALSE_POSITIVE,
                    f"{name} produced an answer for an expected no-answer query.",
                )
            if answerable is True and abstained and context_ids:
                self._add(
                    labels,
                    evidence,
                    FailureType.FALSE_ABSTENTION,
                    f"{name} abstained although expected evidence is in context.",
                )
            if answerable is False and abstained:
                self._add(
                    labels,
                    evidence,
                    FailureType.CORRECT_ABSTENTION,
                    f"{name} abstained on the expected no-answer query.",
                )

    @staticmethod
    def _add(
        labels: list[FailureType],
        evidence: list[str],
        label: FailureType,
        reason: str,
    ) -> None:
        if label not in labels:
            labels.append(label)
            evidence.append(reason)


def _recall(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
