from scripts.validate_sft_dataset import _percentile


def test_percentile_is_deterministic_nearest_rank() -> None:
    values = [1, 2, 3, 4, 5]
    assert _percentile(values, 0.5) == 3
    assert _percentile(values, 0.95) == 5
