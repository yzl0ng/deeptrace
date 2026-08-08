from __future__ import annotations

from scripts.prepare_tool_loop_dev_eval import build_cases


def test_validation_ids_join_to_seed_evidence() -> None:
    cases = build_cases(
        [{"id": "case-1"}],
        [
            {
                "seed_id": "case-1",
                "query": "Question?",
                "language": "en",
                "expected_answer": "Answer",
                "evidence": [
                    {
                        "evidence_id": "e1",
                        "title": "Title",
                        "text": "Evidence",
                    }
                ],
            }
        ],
    )

    assert cases[0]["expected_answer"] == "Answer"
    assert cases[0]["evidence"][0]["content"] == "Evidence"
    assert cases[0]["evidence"][0]["source"] == "frozen-seed:case-1"
    assert cases[0]["gold_claims"][0]["supporting_evidence_ids"] == ["e1"]
