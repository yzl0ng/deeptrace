import math

import pytest

from app.evaluation.retrieval_metrics import (
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_precision_at_k() -> None:
    ranked = ["d1", "d4", "d2"]
    relevance = {"d1": 3, "d2": 1, "d3": 2}

    assert recall_at_k(ranked, relevance, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevance, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, relevance, 5) == pytest.approx(2 / 5)


def test_mrr_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["x", "d2", "d1"], {"d1": 3, "d2": 1}) == 0.5
    assert reciprocal_rank(["x"], {"d1": 3}) == 0.0


def test_dcg_idcg_and_ndcg_support_graded_relevance() -> None:
    relevance = {"d1": 3, "d2": 2, "d3": 1}
    ranked = ["d2", "d1", "x", "d3"]
    expected_dcg = (
        (2**2 - 1) / math.log2(2)
        + (2**3 - 1) / math.log2(3)
        + (2**1 - 1) / math.log2(5)
    )

    assert dcg_at_k([2, 3, 0, 1], 5) == pytest.approx(expected_dcg)
    assert idcg_at_k(relevance, 5) > expected_dcg
    assert ndcg_at_k(["d1", "d2", "d3"], relevance, 5) == 1.0
    assert 0 < ndcg_at_k(ranked, relevance, 5) < 1


def test_no_relevant_documents_returns_zero_metrics() -> None:
    assert recall_at_k(["d1"], {}, 10) == 0.0
    assert precision_at_k(["d1"], {}, 10) == 0.0
    assert reciprocal_rank(["d1"], {}) == 0.0
    assert ndcg_at_k(["d1"], {}, 10) == 0.0


def test_top_k_larger_than_result_count_is_supported() -> None:
    assert recall_at_k(["d1"], {"d1": 1, "d2": 1}, 10) == 0.5
    assert precision_at_k(["d1"], {"d1": 1}, 10) == 0.1


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], {}, 0)
