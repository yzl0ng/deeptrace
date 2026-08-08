from __future__ import annotations

import json

from app.agentic.trajectory import AGENT_SFT_SYSTEM_PROMPT
from scripts.rewrite_sft_prompt_contract import _rewrite_messages


def test_rewrite_messages_preserves_user_and_target() -> None:
    target = json.dumps(
        {
            "steps": [
                {
                    "rationale_summary": "Return the answer.",
                    "action": "answer",
                    "arguments": {},
                    "observation": "",
                    "evidence_ids": [],
                }
            ],
            "final_answer": "answer",
        }
    )
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": target},
    ]

    rewritten = _rewrite_messages(messages)

    assert rewritten[0]["content"] == AGENT_SFT_SYSTEM_PROMPT
    assert rewritten[1] == messages[1]
    assert rewritten[2] == messages[2]
