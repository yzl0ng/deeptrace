from scripts.run_retrieval_evaluation import (
    classify_failures,
    comparison_cases,
    extreme_cases,
    failure_counts,
)


def result(ndcg: float, recall: float) -> dict[str, object]:
    return {
        "ranked_chunk_ids": ["d1"],
        "metrics": {
            "recall@10": recall,
            "ndcg@10": ndcg,
        },
    }


def test_reporting_finds_comparisons_extremes_and_failure_signals() -> None:
    rows = [
        {
            "query_id": "q1",
            "query": "semantic query",
            "category": "semantic-paraphrase",
            "reviewed": True,
            "relevance": {"d1": 1},
            "retrievers": {
                "bm25": result(0.2, 0.0),
                "dense_exact": result(0.8, 1.0),
                "rrf": result(0.9, 1.0),
            },
        },
        {
            "query_id": "q2",
            "query": "RRF",
            "category": "acronym",
            "reviewed": True,
            "relevance": {"d2": 1},
            "retrievers": {
                "bm25": result(1.0, 1.0),
                "dense_exact": result(0.0, 0.0),
                "rrf": result(0.5, 1.0),
            },
        },
    ]

    comparisons = comparison_cases(rows)
    assert comparisons["Dense beats BM25"][0]["query_id"] == "q1"
    assert comparisons["RRF improves over both"][0]["query_id"] == "q1"
    assert extreme_cases(rows)["bm25"]["best"]["query_id"] == "q2"

    failures = classify_failures(rows, [])
    counts = failure_counts(failures)
    assert counts["lexical_mismatch"] == 1
    assert counts["acronym_failure"] == 1
    assert counts["fusion_regression"] == 1
    assert counts["corpus_missing"] == 0
