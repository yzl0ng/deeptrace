from scripts.prepare_trajectory_seed_pool import seed_from_row


def test_hotpot_seed_uses_only_supporting_context() -> None:
    row = {
        "id": "case-1",
        "question": "Which city?",
        "answer": "Paris",
        "context": {
            "title": ["Relevant", "Distractor"],
            "sentences": [["Paris", "is in France."], ["Ignore me."]],
        },
        "supporting_facts": {"title": ["Relevant"], "sent_id": [0]},
    }
    seed = seed_from_row("hotpotqa", row, 0)
    assert seed.seed_id == "hotpotqa-train-case-1"
    assert seed.expected_answer == "Paris"
    assert [item["title"] for item in seed.evidence] == ["Relevant"]


def test_musique_seed_uses_supporting_paragraphs() -> None:
    row = {
        "id": "case-2",
        "question": "Who wrote it?",
        "answer": "An author",
        "paragraphs": [
            {
                "title": "Useful",
                "paragraph_text": "Supporting text",
                "is_supporting": True,
            },
            {
                "title": "Noise",
                "paragraph_text": "Distractor text",
                "is_supporting": False,
            },
        ],
    }
    seed = seed_from_row("musique", row, 1)
    assert seed.variant == "verification"
    assert [item["title"] for item in seed.evidence] == ["Useful"]
