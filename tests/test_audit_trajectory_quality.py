from app.agentic.trajectory import TrajectorySeed
from scripts.audit_trajectory_quality import (
    audit_records,
    normalize_answer,
    token_f1,
)


def _seed() -> TrajectorySeed:
    return TrajectorySeed(
        seed_id="hotpotqa-train-1",
        query="Who wrote the work?",
        language="en",
        variant="research",
        expected_answer="The Example Author",
        evidence=[
            {
                "evidence_id": "e1",
                "title": "Example",
                "text": "The work was written by Example Author.",
            }
        ],
    )


def test_answer_normalization_and_token_f1() -> None:
    assert normalize_answer("  THE Example, Author! ") == "example author"
    assert token_f1("Example Author", "The Example Author") == 1.0


def test_audit_records_passes_a_well_formed_trajectory() -> None:
    seed = _seed()
    text = """
    {
      "seed_id": "hotpotqa-train-1",
      "query": "Who wrote the work?",
      "variant": "research",
      "steps": [
        {
          "rationale_summary": "Find the supplied evidence.",
          "action": "search",
          "arguments": {"query": "work author"},
          "observation": "Found the Example page.",
          "evidence_ids": ["e1"]
        },
        {
          "rationale_summary": "The evidence directly names the author.",
          "action": "evaluate_evidence",
          "arguments": {},
          "observation": "The answer is supported.",
          "evidence_ids": ["e1"]
        },
        {
          "rationale_summary": "Return the supported answer.",
          "action": "answer",
          "arguments": {"answer": "Example Author"},
          "observation": "",
          "evidence_ids": ["e1"]
        }
      ],
      "final_answer": "Example Author"
    }
    """
    metrics, rows = audit_records(
        [{"seed_id": seed.seed_id, "text": text}],
        {seed.seed_id: seed},
    )

    assert len(rows) == 1
    assert metrics["accepted"]["normalized_exact_rate"] == 1.0
    assert metrics["accepted"]["final_supported_by_seed_evidence_rate"] == 1.0
    assert (
        metrics["quality_gate"]["checks"][
            "normalized_exact_at_least_99_percent"
        ]
        is True
    )
    assert (
        metrics["quality_gate"]["checks"]["accepted_at_least_400_records"]
        is False
    )


def test_audit_records_counts_filter_rejections() -> None:
    seed = _seed()
    metrics, rows = audit_records(
        [{"seed_id": seed.seed_id, "text": "not json"}],
        {seed.seed_id: seed},
    )

    assert rows == []
    assert metrics["rejected_records"] == 1
    assert metrics["rejection_reasons"] == {"invalid_json": 1}
    assert metrics["quality_gate"]["passed"] is False
