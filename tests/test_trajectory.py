from __future__ import annotations

import json
from pathlib import Path

from app.agentic.trajectory import (
    AGENT_SFT_SYSTEM_PROMPT,
    TrajectorySeed,
    build_pilot_seeds,
    parse_and_filter_trajectory,
    parse_sft_response,
    to_sft_record,
)


def test_pilot_seed_builder_produces_one_hundred_unique_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    seeds = build_pilot_seeds(root)

    assert len(seeds) == 100
    assert len({seed.seed_id for seed in seeds}) == 100


def test_valid_teacher_json_becomes_sft_record() -> None:
    seed = TrajectorySeed(
        seed_id="case-1",
        query="What is grounded search?",
        language="en",
        variant="research",
        evidence=[{"evidence_id": "e1", "content": "Grounded evidence."}],
    )
    payload = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "Check the supplied evidence.",
                "action": "evaluate_evidence",
                "arguments": {},
                "observation": "The evidence supports the answer.",
                "evidence_ids": ["e1"],
            },
            {
                "rationale_summary": "Return the grounded answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": ["e1"],
            },
        ],
        "final_answer": "Grounded search uses retrieved evidence.",
    }

    trajectory, reasons = parse_and_filter_trajectory(
        json.dumps(payload),
        seed,
    )

    assert reasons == []
    assert trajectory is not None
    record = to_sft_record(trajectory)
    assert record["id"] == "case-1"
    assert len(record["messages"]) == 3
    assert record["messages"][0]["content"] == AGENT_SFT_SYSTEM_PROMPT
    assert "Return exactly one JSON object" in AGENT_SFT_SYSTEM_PROMPT
    assert '"steps"' in AGENT_SFT_SYSTEM_PROMPT
    assert '"final_answer"' in AGENT_SFT_SYSTEM_PROMPT


def test_invented_evidence_id_is_rejected() -> None:
    seed = TrajectorySeed(
        seed_id="case-2",
        query="Question",
        language="en",
        variant="verification",
        evidence=[{"evidence_id": "known", "content": "Evidence"}],
    )
    payload = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "Evaluate evidence.",
                "action": "evaluate_evidence",
                "arguments": {},
                "observation": "Observed.",
                "evidence_ids": ["invented"],
            },
            {
                "rationale_summary": "Return the answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": [],
            },
        ],
        "final_answer": "Answer",
    }

    trajectory, reasons = parse_and_filter_trajectory(
        json.dumps(payload),
        seed,
    )

    assert trajectory is None
    assert reasons == ["invented_evidence_id"]


def test_answer_step_may_leave_observation_empty() -> None:
    seed = TrajectorySeed(
        seed_id="case-3",
        query="Question",
        language="en",
        variant="research",
    )
    payload = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "Return the final answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": [],
            }
        ],
        "final_answer": "Answer",
    }

    trajectory, reasons = parse_and_filter_trajectory(
        json.dumps(payload),
        seed,
    )

    assert trajectory is not None
    assert reasons == []


def test_schema_error_reports_field_and_type() -> None:
    seed = TrajectorySeed(
        seed_id="case-4",
        query="Question",
        language="en",
        variant="research",
    )
    payload = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "Search first.",
                "action": "search",
                "arguments": {},
                "observation": "",
                "evidence_ids": [],
            }
        ],
        "final_answer": "Answer",
    }

    trajectory, reasons = parse_and_filter_trajectory(
        json.dumps(payload),
        seed,
    )

    assert trajectory is None
    assert reasons == [
        "invalid_schema:steps.0:value_error"
    ]


def test_short_benchmark_answer_is_valid() -> None:
    seed = TrajectorySeed(
        seed_id="case-5",
        query="Is the claim true?",
        language="en",
        variant="verification",
    )
    payload = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "Return the verified answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": [],
            }
        ],
        "final_answer": "No",
    }

    trajectory, reasons = parse_and_filter_trajectory(
        json.dumps(payload),
        seed,
    )

    assert trajectory is not None
    assert reasons == []


def test_parse_sft_response_accepts_training_output_shape() -> None:
    payload = {
        "steps": [
            {
                "rationale_summary": "Return the grounded answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": [],
            }
        ],
        "final_answer": "Answer",
    }

    parsed = parse_sft_response(json.dumps(payload))

    assert parsed is not None
    assert parsed.steps[0].action == "answer"


def test_filter_rejects_a_trajectory_without_final_answer_action() -> None:
    seed = TrajectorySeed(
        seed_id="seed-no-answer",
        query="Who wrote it?",
        language="en",
        variant="research",
        expected_answer="Example Author",
        evidence=[],
    )
    text = """
    {
      "seed_id": "seed-no-answer",
      "query": "Who wrote it?",
      "variant": "research",
      "steps": [{
        "rationale_summary": "The evidence is sufficient.",
        "action": "evaluate_evidence",
        "arguments": {},
        "observation": "The author is known.",
        "evidence_ids": []
      }],
      "final_answer": "Example Author"
    }
    """

    trajectory, reasons = parse_and_filter_trajectory(text, seed)

    assert trajectory is None
    assert reasons == [
        "missing_answer_action",
        "answer_action_not_final",
    ]
