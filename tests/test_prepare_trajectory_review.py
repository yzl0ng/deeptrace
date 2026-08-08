from scripts.prepare_trajectory_review import _sample_key, _source


def test_review_source_comes_from_seed_prefix() -> None:
    assert _source("hotpotqa-train-abc") == "hotpotqa"
    assert _source("musique-train-123") == "musique"


def test_review_sample_key_is_deterministic() -> None:
    assert _sample_key("case-1", "seed") == _sample_key("case-1", "seed")
    assert _sample_key("case-1", "seed") != _sample_key("case-2", "seed")
