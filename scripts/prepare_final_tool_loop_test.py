from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

SOURCES = (
    {
        "name": "hotpotqa",
        "repository": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "license": "CC-BY-SA-4.0",
    },
    {
        "name": "2wikimultihopqa",
        "repository": "framolfese/2WikiMultihopQA",
        "config": "default",
        "split": "validation",
        "revision": "fe713bfbd1afbca1a65246741a75890405d56a3a",
        "license": "Apache-2.0",
    },
    {
        "name": "musique",
        "repository": "bdsaglam/musique",
        "config": "answerable",
        "split": "validation",
        "revision": "22873a405dd809893b22ada0b499299fb612d2df",
        "license": "CC-BY-4.0",
    },
)


def _normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _selection_key(seed: str, source: str, record_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{source}:{record_id}".encode("utf-8")
    ).hexdigest()


def _excluded_questions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if question:
                excluded.add(_normalize_question(question))
    return excluded


def _rows_from_dataset_server(
    source: dict[str, str],
    *,
    skip_first: int,
    candidate_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while len(rows) < candidate_window:
        length = min(100, candidate_window - len(rows))
        query = urlencode(
            {
                "dataset": source["repository"],
                "config": source["config"],
                "split": source["split"],
                "offset": skip_first + len(rows),
                "length": length,
                "revision": source["revision"],
            }
        )
        with urlopen(f"{ROWS_ENDPOINT}?{query}", timeout=60) as response:
            payload = json.loads(response.read())
        page = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page)
        if len(page) < length:
            break
    return rows


def _hotpot_shape(
    source: dict[str, str],
    row: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    record_id = str(row.get("id") or row.get("_id"))
    context = row.get("context", {})
    titles = list(context.get("title", []))
    sentences = list(context.get("sentences", []))
    supporting = row.get("supporting_facts", {})
    support_titles = {
        str(title) for title in supporting.get("title", [])
    }
    evidence: list[dict[str, str]] = []
    gold_ids: list[str] = []
    for index, (title, parts) in enumerate(
        zip(titles, sentences, strict=True)
    ):
        evidence_id = f"{source['name']}-{record_id}-context-{index}"
        text = " ".join(str(part) for part in parts).strip()
        if not text:
            continue
        evidence.append(
            {
                "evidence_id": evidence_id,
                "title": str(title),
                "content": text[:3000],
                "source": f"{source['repository']}:{record_id}",
            }
        )
        if str(title) in support_titles:
            gold_ids.append(evidence_id)
    return evidence, list(dict.fromkeys(gold_ids))


def _musique_shape(
    source: dict[str, str],
    row: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    record_id = str(row.get("id") or row.get("_id"))
    evidence: list[dict[str, str]] = []
    gold_ids: list[str] = []
    for index, paragraph in enumerate(row.get("paragraphs", [])):
        text = str(paragraph.get("paragraph_text", "")).strip()
        if not text:
            continue
        evidence_id = f"musique-{record_id}-context-{index}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "title": str(paragraph.get("title", "")),
                "content": text[:3000],
                "source": f"{source['repository']}:{record_id}",
            }
        )
        if bool(paragraph.get("is_supporting", False)):
            gold_ids.append(evidence_id)
    return evidence, list(dict.fromkeys(gold_ids))


def build_case_variants(
    source: dict[str, str],
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    record_id = str(row.get("id") or row.get("_id") or "").strip()
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not record_id or not question or not answer:
        return None
    if source["name"] == "musique":
        all_evidence, gold_ids = _musique_shape(source, row)
        aliases = [
            str(item) for item in row.get("answer_aliases", [])
            if str(item).strip()
        ]
    else:
        all_evidence, gold_ids = _hotpot_shape(source, row)
        aliases = []
    if len(gold_ids) < 2:
        return None
    evidence_by_id = {
        item["evidence_id"]: item for item in all_evidence
    }
    controlled_evidence = [
        evidence_by_id[evidence_id] for evidence_id in gold_ids
    ]
    case_id = f"{source['name']}-validation-{record_id}"
    common = {
        "case_id": case_id,
        "dataset": source["name"],
        "split": "test",
        "language": "en",
        "question": question,
        "expected_answer": answer,
        "answer_aliases": aliases,
        "gold_claims": [
            {
                "claim_id": f"{case_id}-gold-answer",
                "text": f'The answer to "{question}" is "{answer}".',
                "supporting_evidence_ids": gold_ids,
                "contradicting_evidence_ids": [],
            }
        ],
        "reviewed": False,
        "source_record_id": record_id,
    }
    controlled = {**common, "evidence": controlled_evidence}
    distractor = {**common, "evidence": all_evidence}
    return controlled, distractor


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced, pinned, unseen final tool-loop test with "
            "controlled and distractor evidence variants."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-source", type=int, default=30)
    parser.add_argument("--candidate-window", type=int, default=500)
    parser.add_argument("--skip-first", type=int, default=2)
    parser.add_argument(
        "--selection-seed",
        default="deeptrace-final-test-20260730-v1",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--transport",
        choices=("datasets", "datasets-server"),
        default="datasets",
    )
    args = parser.parse_args()
    if args.records_per_source < 1 or args.candidate_window < 1:
        parser.error("record counts must be positive")
    if args.skip_first < 0:
        parser.error("skip-first must be non-negative")

    if args.transport == "datasets":
        from datasets import load_dataset

    excluded = _excluded_questions(args.exclude_jsonl)
    selected_controlled: list[dict[str, Any]] = []
    selected_distractor: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in SOURCES:
        if args.transport == "datasets":
            dataset_rows: Iterable[dict[str, Any]] = islice(
                load_dataset(
                    source["repository"],
                    source["config"],
                    split=source["split"],
                    revision=source["revision"],
                    streaming=True,
                ),
                args.skip_first,
                args.skip_first + args.candidate_window,
            )
        else:
            dataset_rows = _rows_from_dataset_server(
                source,
                skip_first=args.skip_first,
                candidate_window=args.candidate_window,
            )
        candidates: list[
            tuple[str, dict[str, Any], dict[str, Any]]
        ] = []
        for row in dataset_rows:
            variants = build_case_variants(source, row)
            if variants is None:
                continue
            controlled, distractor = variants
            if _normalize_question(controlled["question"]) in excluded:
                continue
            key = _selection_key(
                args.selection_seed,
                source["name"],
                controlled["source_record_id"],
            )
            candidates.append((key, controlled, distractor))
        candidates.sort(key=lambda item: item[0])
        chosen = candidates[: args.records_per_source]
        if len(chosen) != args.records_per_source:
            raise RuntimeError(
                f"{source['name']} produced only {len(chosen)} valid cases"
            )
        selected_controlled.extend(item[1] for item in chosen)
        selected_distractor.extend(item[2] for item in chosen)
        source_summaries.append(
            {
                **source,
                "candidate_window": args.candidate_window,
                "valid_candidates": len(candidates),
                "selected": len(chosen),
            }
        )

    case_ids = [row["case_id"] for row in selected_controlled]
    questions = [row["question"] for row in selected_controlled]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("final test case IDs are not unique")
    if len({_normalize_question(item) for item in questions}) != len(
        questions
    ):
        raise RuntimeError("final test questions are not unique")
    if {_normalize_question(item) for item in questions} & excluded:
        raise RuntimeError("final test overlaps an excluded question")

    controlled_bytes = _jsonl(selected_controlled)
    distractor_bytes = _jsonl(selected_distractor)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    controlled_path = args.output_dir / "test-controlled.jsonl"
    distractor_path = args.output_dir / "test-distractor.jsonl"
    controlled_path.write_bytes(controlled_bytes)
    distractor_path.write_bytes(distractor_bytes)
    manifest = {
        "dataset_version": "final-tool-loop-test-v1",
        "status": "frozen_before_evaluation",
        "formal_split": "test",
        "cases": len(selected_controlled),
        "selection_seed": args.selection_seed,
        "transport": args.transport,
        "skip_first": args.skip_first,
        "excluded_question_count": len(excluded),
        "sources": source_summaries,
        "artifacts": {
            controlled_path.name: {
                "bytes": len(controlled_bytes),
                "sha256": _sha256_bytes(controlled_bytes),
                "evidence_scope": "gold supporting evidence only",
            },
            distractor_path.name: {
                "bytes": len(distractor_bytes),
                "sha256": _sha256_bytes(distractor_bytes),
                "evidence_scope": (
                    "full upstream context including distractors"
                ),
            },
        },
        "overlap_checks": {
            "excluded_question_overlap": 0,
            "unique_case_ids": len(set(case_ids)),
            "unique_questions": len(
                {_normalize_question(item) for item in questions}
            ),
        },
        "metric_policy": {
            "answer": "normalized exact match against answer or alias",
            "evidence_recall": (
                "gold evidence IDs read or cited divided by all gold IDs, "
                "macro-averaged over cases"
            ),
            "completion": "valid answer action within eight tool steps",
        },
        "tuning_policy": (
            "Do not change prompts, protocol gates, model weights, or "
            "thresholds after observing these test results."
        ),
        "truth_boundary": (
            "Labels and supporting facts come from pinned public validation "
            "splits. They are benchmark gold, not new human annotation."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
