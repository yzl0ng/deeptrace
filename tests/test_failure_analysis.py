from app.evaluation.failure_analysis import FailureAnalyzer, FailureType
from scripts.run_failure_analysis import (
    render_report,
    write_json,
    write_jsonl,
)


def case(
    *,
    bm25: list[str] | None = None,
    dense: list[str] | None = None,
    rrf: list[str] | None = None,
    before: list[str] | None = None,
    after: list[str] | None = None,
    relevant: dict[str, int] | None = None,
    answerable: bool | None = True,
    rag: dict | None = None,
    case_id: str = "case-1",
) -> dict:
    expected = {"answerable": answerable}
    if relevant is not None:
        expected["relevance"] = relevant
    return {
        "case_id": case_id,
        "query_id": case_id,
        "query": "test query",
        "expected": expected,
        "retrieval": {
            "bm25": {"ranked_chunk_ids": bm25 or []},
            "dense": {"ranked_chunk_ids": dense or []},
            "rrf": {"ranked_chunk_ids": rrf or []},
            "reranker": {
                "before": before or [],
                "after": after or [],
            },
        },
        "rag": rag or {},
        "reviewed": False,
    }


def labels(value: dict) -> set[str]:
    return set(FailureAnalyzer().analyze_query(value)["failure_types"])


def test_classifies_dense_beating_bm25_as_lexical_mismatch() -> None:
    value = case(
        bm25=["x"],
        dense=["relevant"],
        rrf=["relevant"],
        relevant={"relevant": 1},
    )
    assert FailureType.LEXICAL_MISMATCH in labels(value)


def test_classifies_bm25_beating_dense_as_dense_false_positive() -> None:
    value = case(
        bm25=["relevant"],
        dense=["x"],
        rrf=["relevant"],
        relevant={"relevant": 1},
    )
    assert FailureType.DENSE_FALSE_POSITIVE in labels(value)


def test_classifies_rrf_improvement_and_regression() -> None:
    improvement = case(
        bm25=["x", "relevant"],
        dense=["y", "relevant"],
        rrf=["relevant"],
        relevant={"relevant": 1},
    )
    regression = case(
        bm25=["relevant"],
        dense=["relevant"],
        rrf=["x", "relevant"],
        relevant={"relevant": 1},
    )
    assert FailureType.FUSION_IMPROVEMENT in labels(improvement)
    assert FailureType.FUSION_REGRESSION in labels(regression)


def test_classifies_reranker_improvement_regression_and_missing() -> None:
    common = {
        "bm25": ["relevant"],
        "dense": ["relevant"],
        "rrf": ["relevant"],
        "relevant": {"relevant": 1},
    }
    assert FailureType.RERANKER_IMPROVEMENT in labels(
        case(**common, before=["x", "relevant"], after=["relevant", "x"])
    )
    assert FailureType.RERANKER_REGRESSION in labels(
        case(**common, before=["relevant", "x"], after=["x", "relevant"])
    )
    assert FailureType.CANDIDATE_MISSING in labels(
        case(**common, before=["x"], after=["x"])
    )


def test_classifies_citation_and_abstention_signals() -> None:
    rag = {
        "baseline": {
            "answer": "answer",
            "context_chunk_ids": ["doc-1"],
            "citations": ["doc-2"],
            "invalid_citation_ids": ["doc-2"],
            "abstained": False,
        }
    }
    assert FailureType.CITATION_INVALID_ID in labels(
        case(relevant={"doc-1": 1}, rag=rag)
    )
    missing = {
        "baseline": {
            "answer": "answer",
            "context_chunk_ids": ["doc-1"],
            "citations": [],
            "invalid_citation_ids": [],
            "abstained": False,
        }
    }
    assert FailureType.CITATION_MISSING in labels(
        case(relevant={"doc-1": 1}, rag=missing)
    )
    assert FailureType.NO_ANSWER_FALSE_POSITIVE in labels(
        case(relevant={}, answerable=False, rag=missing)
    )
    false_abstention = {
        "baseline": {
            "answer": "cannot answer",
            "context_chunk_ids": ["doc-1"],
            "citations": [],
            "invalid_citation_ids": [],
            "abstained": True,
        }
    }
    assert FailureType.FALSE_ABSTENTION in labels(
        case(relevant={"doc-1": 1}, rag=false_abstention)
    )


def test_marks_missing_annotations_uncertain() -> None:
    assert FailureType.ANNOTATION_UNCERTAIN in labels(
        case(relevant=None, answerable=None)
    )


def test_deduplicates_cases_and_is_deterministic() -> None:
    analyzer = FailureAnalyzer()
    first = case(case_id="same", relevant={"doc": 1})
    second = case(case_id="same", relevant={"other": 1})
    output_a = analyzer.analyze_many([first, second])
    output_b = analyzer.analyze_many([first, second])
    assert output_a == output_b
    assert len(output_a) == 1


def test_summary_separates_review_status_and_aggregates() -> None:
    analyzer = FailureAnalyzer()
    first = analyzer.analyze_query(
        case(
            case_id="one",
            bm25=["x"],
            dense=["doc"],
            rrf=["doc"],
            relevant={"doc": 1},
        )
    )
    second = analyzer.analyze_query(
        case(
            case_id="two",
            bm25=["doc"],
            dense=["x"],
            rrf=["doc"],
            relevant={"doc": 1},
        )
    )
    second["reviewed"] = True
    summary = analyzer.summarize(
        [first, second],
        experiment_id="test",
        generated_at="2026-01-01T00:00:00Z",
        corpus_version="fixture",
        query_set_version="fixture-v1",
    )
    assert summary["queries_analyzed"] == 2
    assert summary["reviewed_cases"] == 1
    assert summary["unreviewed_candidates"] == 1
    assert summary["failure_counts"]["lexical_mismatch"] == 1
    assert summary["pipeline_stage_counts"]["bm25"] == 1
    assert "prompt_injection_risk" in summary["no_observed_case"]


def test_rag_comparison_reports_changes_without_quality_claim() -> None:
    comparison = FailureAnalyzer().compare_rag_variants(
        {
            "baseline": {
                "answer": "a",
                "citations": ["doc-1"],
                "abstained": False,
                "context_chunk_ids": ["doc-1"],
                "latency": {"total_ms": 1},
                "usage": {"total_tokens": 10},
            },
            "reranked": {
                "answer": "b",
                "citations": ["doc-2"],
                "abstained": True,
                "context_chunk_ids": ["doc-2"],
                "latency": {"total_ms": 2},
                "usage": {"total_tokens": 12},
            },
        }
    )
    assert comparison["answer_changed"] is True
    assert comparison["context_changed"] is True
    assert comparison["cost"] is None


def test_writes_config_jsonl_and_markdown_report(tmp_path) -> None:
    analyzer = FailureAnalyzer()
    candidate = analyzer.analyze_query(
        case(
            bm25=["doc"],
            dense=["x"],
            rrf=["doc"],
            relevant={"doc": 1},
        )
    )
    summary = analyzer.summarize(
        [candidate],
        experiment_id="fixture",
        generated_at="2026-01-01T00:00:00Z",
        corpus_version="fixture",
        query_set_version="fixture",
    )
    config = {
        "generated_at": "2026-01-01T00:00:00Z",
        "inputs": {
            "retrieval_per_query": "saved.jsonl",
            "baseline_rag_v1": "baseline.jsonl",
            "live_rag_comparison": "comparison.jsonl",
        },
        "pipelines": {
            "baseline": "RRF Top 5 -> DeepSeek",
            "reranked": "RRF Top 20 -> Reranker Top 5 -> DeepSeek",
        },
    }
    write_json(tmp_path / "config.json", config)
    write_jsonl(tmp_path / "cases.jsonl", [candidate])
    report = render_report(config, summary, [candidate])
    (tmp_path / "report.md").write_text(report, encoding="utf-8")
    assert '"generated_at": "2026-01-01T00:00:00Z"' in (
        tmp_path / "config.json"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "cases.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 1
    assert "自动标签只用于定位人工复核候选" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")
