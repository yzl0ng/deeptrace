import pytest

from app.evaluation.dataset_sampling import (
    filter_qrels,
    select_document_ids,
    select_query_ids,
)


QRELS = [
    ("q1", "d1", 1),
    ("q1", "d2", 1),
    ("q2", "d3", 1),
    ("q3", "d4", 0),
    ("q4", "d5", 2),
]


def test_query_sampling_is_deterministic_and_positive_only() -> None:
    first = select_query_ids(QRELS, count=2, seed=7)
    second = select_query_ids(QRELS, count=2, seed=7)

    assert first == second
    assert set(first) <= {"q1", "q2", "q4"}
    assert "q3" not in first


def test_document_sampling_keeps_every_judged_document() -> None:
    selected = select_document_ids(
        ["d1", "d2", "d3", "d4", "d5", "d6"],
        required_ids={"d1", "d5"},
        count=4,
        seed=11,
    )

    assert len(selected) == 4
    assert {"d1", "d5"} <= set(selected)
    assert selected == sorted(selected)


def test_document_sampling_rejects_impossible_profiles() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        select_document_ids(
            ["d1", "d2"],
            required_ids={"d1", "d2"},
            count=1,
            seed=1,
        )

    with pytest.raises(ValueError, match="missing"):
        select_document_ids(
            ["d1"],
            required_ids={"missing"},
            count=1,
            seed=1,
        )


def test_filter_qrels_preserves_only_selected_positive_labels() -> None:
    filtered = filter_qrels(
        QRELS,
        query_ids={"q1", "q3"},
        document_ids={"d1", "d4"},
    )

    assert filtered == [("q1", "d1", 1)]
