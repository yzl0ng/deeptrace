from __future__ import annotations

from scripts.prepare_final_tool_loop_test import build_case_variants


def test_build_case_variants_separates_gold_and_distractors() -> None:
    source = {
        "name": "hotpotqa",
        "repository": "fixture/hotpot",
        "config": "distractor",
        "split": "validation",
        "revision": "a" * 40,
        "license": "CC-BY-SA-4.0",
    }
    row = {
        "id": "fixture-1",
        "question": "Which year did the director's film open?",
        "answer": "1999",
        "context": {
            "title": ["Film", "Director", "Distractor"],
            "sentences": [
                ["The film was directed by Alex."],
                ["Alex's next film opened in 1999."],
                ["This paragraph is unrelated."],
            ],
        },
        "supporting_facts": {
            "title": ["Film", "Director"],
            "sent_id": [0, 0],
        },
    }

    variants = build_case_variants(source, row)

    assert variants is not None
    controlled, distractor = variants
    assert len(controlled["evidence"]) == 2
    assert len(distractor["evidence"]) == 3
    assert len(
        controlled["gold_claims"][0]["supporting_evidence_ids"]
    ) == 2
