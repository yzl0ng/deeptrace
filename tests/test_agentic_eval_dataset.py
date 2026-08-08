from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from app.evaluation.agentic import AgentEvalCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agentic-search-v1"
)


def _load_jsonl(path: Path) -> list[AgentEvalCase]:
    return [
        AgentEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_frozen_public_subset_matches_manifest_and_source_balance() -> None:
    test_path = DATASET_DIR / "test.jsonl"
    manifest = json.loads(
        (DATASET_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    cases = _load_jsonl(test_path)

    assert digest == manifest["test_sha256"]
    assert manifest["formal_split"] == "test"
    assert manifest["test_cases"] == 6
    assert manifest["tuning_policy"]
    assert Counter(item.dataset for item in cases) == {
        "hotpotqa": 2,
        "2wikimultihopqa": 2,
        "musique": 2,
    }
    assert all(item.split == "test" and item.reviewed for item in cases)


def test_every_gold_claim_maps_to_existing_multi_hop_evidence() -> None:
    for case in _load_jsonl(DATASET_DIR / "test.jsonl"):
        evidence_ids = {item.evidence_id for item in case.evidence}
        assert len(evidence_ids) >= 2
        for claim in case.gold_claims:
            assert len(claim.supporting_evidence_ids) >= 2
            assert set(claim.supporting_evidence_ids) <= evidence_ids


def test_dataset_revisions_and_licenses_are_pinned() -> None:
    manifest = json.loads(
        (DATASET_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert {item["license"] for item in manifest["sources"]} == {
        "CC-BY-SA-4.0",
        "Apache-2.0",
        "CC-BY-4.0",
    }
    assert all(
        len(item["revision"]) == 40 for item in manifest["sources"]
    )


def test_chinese_draft_is_excluded_from_formal_metrics() -> None:
    cases = _load_jsonl(DATASET_DIR / "chinese-draft.jsonl")
    assert len(cases) == 2
    assert all(item.language == "zh" for item in cases)
    assert all(item.split == "draft" for item in cases)
    assert all(not item.reviewed for item in cases)
