import json
from pathlib import Path

from scripts.prepare_sft_pilot import _normalize_question


def test_normalize_question_collapses_spacing_and_case() -> None:
    assert _normalize_question("  What  IS this? ") == "what is this?"


def test_forbidden_question_fixture_is_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"question": "A frozen question?"}) + "\n",
        encoding="utf-8",
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert _normalize_question(rows[0]["question"]) == "a frozen question?"
