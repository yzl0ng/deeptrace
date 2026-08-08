from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agentic-search-v1"
)
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASETS = (
    {
        "name": "hotpotqa",
        "repository": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "license": "CC-BY-SA-4.0",
        "upstream": "https://github.com/hotpotqa/hotpot",
    },
    {
        "name": "2wikimultihopqa",
        "repository": "framolfese/2WikiMultihopQA",
        "config": "default",
        "split": "validation",
        "license": "Apache-2.0",
        "upstream": "https://github.com/Alab-NII/2wikimultihop",
    },
    {
        "name": "musique",
        "repository": "bdsaglam/musique",
        "config": "answerable",
        "split": "validation",
        "license": "CC-BY-4.0",
        "upstream": "https://github.com/stonybrooknlp/musique",
    },
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    with httpx.Client(timeout=60) as client:
        for specification in DATASETS:
            repository = str(specification["repository"])
            metadata = client.get(
                f"https://huggingface.co/api/datasets/{repository}"
            )
            metadata.raise_for_status()
            revision = str(metadata.json()["sha"])
            response = client.get(
                ROWS_ENDPOINT,
                params={
                    "dataset": repository,
                    "config": specification["config"],
                    "split": specification["split"],
                    "offset": 0,
                    "length": 2,
                    "revision": revision,
                },
            )
            response.raise_for_status()
            rows = response.json()["rows"]
            if len(rows) != 2:
                raise RuntimeError(
                    f"expected exactly two rows for {repository}"
                )
            sources.append(
                {
                    **specification,
                    "revision": revision,
                    "offset": 0,
                    "length": 2,
                }
            )
            for row in rows:
                raw = row["row"]
                if specification["name"] == "musique":
                    cases.append(_from_musique(raw))
                else:
                    cases.append(
                        _from_hotpot_shape(
                            raw,
                            dataset=str(specification["name"]),
                            repository=repository,
                        )
                    )

    test_path = OUTPUT_DIR / "test.jsonl"
    test_path.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": "agentic-search-v1",
        "status": "frozen",
        "formal_split": "test",
        "selection": (
            "First two validation rows at each pinned repository revision; "
            "selection and verifier thresholds were fixed before evaluation."
        ),
        "test_cases": len(cases),
        "test_sha256": _sha256(test_path),
        "sources": sources,
        "tuning_policy": (
            "The test split must not be used to change verifier thresholds "
            "or scripted mode policies."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _from_hotpot_shape(
    raw: dict[str, Any],
    *,
    dataset: str,
    repository: str,
) -> dict[str, Any]:
    titles = raw["context"]["title"]
    sentences = raw["context"]["sentences"]
    context = {
        str(title): list(items)
        for title, items in zip(titles, sentences, strict=True)
    }
    evidence: list[dict[str, str]] = []
    for index, (title, sentence_id) in enumerate(
        zip(
            raw["supporting_facts"]["title"],
            raw["supporting_facts"]["sent_id"],
            strict=True,
        ),
        start=1,
    ):
        sentence = str(context[str(title)][int(sentence_id)]).strip()
        evidence.append(
            {
                "evidence_id": (
                    f"{dataset}-{raw['id']}-support-{index}"
                ),
                "title": str(title),
                "content": sentence,
                "source": f"{repository}:{raw['id']}",
            }
        )
    return _case(raw, dataset=dataset, evidence=evidence)


def _from_musique(raw: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": (
                f"musique-{raw['id']}-support-{index + 1}"
            ),
            "title": str(item["title"]),
            "content": str(item["paragraph_text"]).strip(),
            "source": f"bdsaglam/musique:{raw['id']}",
        }
        for index, item in enumerate(raw["paragraphs"])
        if item["is_supporting"]
    ]
    return _case(
        raw,
        dataset="musique",
        evidence=evidence,
        aliases=list(raw.get("answer_aliases", [])),
    )


def _case(
    raw: dict[str, Any],
    *,
    dataset: str,
    evidence: list[dict[str, str]],
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    if len(evidence) < 2:
        raise RuntimeError(
            f"{dataset}:{raw['id']} does not contain multi-hop evidence"
        )
    question = str(raw["question"])
    answer = str(raw["answer"])
    case_id = f"{dataset}-{raw['id']}"
    return {
        "case_id": case_id,
        "dataset": dataset,
        "split": "test",
        "language": "en",
        "question": question,
        "expected_answer": answer,
        "answer_aliases": aliases or [],
        "evidence": evidence,
        "gold_claims": [
            {
                "claim_id": f"{case_id}-gold-answer",
                "text": f'The answer to "{question}" is "{answer}".',
                "supporting_evidence_ids": [
                    item["evidence_id"] for item in evidence
                ],
                "contradicting_evidence_ids": [],
            }
        ],
        "reviewed": True,
        "source_record_id": str(raw["id"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
