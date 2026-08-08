import json

from scripts.evaluate_lora_tool_format import _normalize_answer
from scripts.prepare_sft_dev_eval import build_dev_cases


def test_build_dev_cases_extracts_question_and_final_answer() -> None:
    rows = [
        {
            "id": "case-1",
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Who wrote it?"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"steps": [], "final_answer": "Example Author"}
                    ),
                },
            ],
        }
    ]

    cases = build_dev_cases(rows)

    assert cases == [
        {
            "case_id": "case-1",
            "question": "Who wrote it?",
            "expected_answer": "Example Author",
            "split": "validation_dev",
        }
    ]


def test_eval_answer_normalization_ignores_articles_and_punctuation() -> None:
    assert _normalize_answer("The Example, Author!") == "example author"
