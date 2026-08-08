from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_upstreams_are_pinned_to_real_commits() -> None:
    lock_text = (ROOT / "upstream.lock.yaml").read_text(encoding="utf-8")

    for component in (
        "open-deep-research",
        "agent-r1",
        "verl",
        "search-r1",
        "gpt-researcher",
        "tongyi-deepresearch",
    ):
        assert f"name: {component}" in lock_text
    commits = re.findall(r"^\s+commit:\s+([0-9a-f]+)$", lock_text, re.MULTILINE)
    assert len(commits) == 6
    assert all(len(commit) == 40 for commit in commits)
    assert "<" not in lock_text
    assert "runtime_validated: false" in lock_text


def test_provenance_files_do_not_claim_source_was_copied() -> None:
    third_party = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "No source files copied" in third_party
    assert "has not copied source files" in notice
