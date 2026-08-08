from __future__ import annotations

from scripts.evaluate_lora_tool_format import _normalize_answer


def test_answer_normalization_is_case_and_article_insensitive() -> None:
    assert _normalize_answer("The Mask of Fu Manchu") == (
        _normalize_answer("mask of fu manchu")
    )
