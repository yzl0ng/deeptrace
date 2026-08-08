from __future__ import annotations

import json

from scripts.prepare_action_sft_dataset import convert_messages


def test_full_trajectory_becomes_action_and_tool_turns() -> None:
    trajectory = {
        "steps": [
            {
                "rationale_summary": "Search for evidence.",
                "action": "search",
                "arguments": {"query": "RRF"},
                "observation": "Found evidence e1.",
                "evidence_ids": ["e1"],
            },
            {
                "rationale_summary": "Return the answer.",
                "action": "answer",
                "arguments": {},
                "observation": "",
                "evidence_ids": ["e1"],
            },
        ],
        "final_answer": "RRF",
    }
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "What combines ranked lists?"},
        {"role": "assistant", "content": json.dumps(trajectory)},
    ]

    converted = convert_messages(messages)

    assert [item["role"] for item in converted] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    final_action = json.loads(converted[-1]["content"])
    assert final_action["action"] == "answer"
    assert final_action["final_answer"] == "RRF"
