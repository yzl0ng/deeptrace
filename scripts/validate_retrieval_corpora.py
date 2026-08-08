from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated SearchLab retrieval corpus profiles."
    )
    parser.add_argument(
        "profiles",
        nargs="*",
        type=Path,
        default=[
            Path("data/experiments/scifact_en/quality"),
            Path("data/experiments/trec_covid_en/scale_50k"),
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for profile in args.profiles:
        summary = validate_profile(profile)
        print(
            f"OK {profile}: {summary['documents']} documents, "
            f"{summary['queries']} queries, {summary['qrels']} qrels"
        )


def validate_profile(profile: Path) -> dict[str, int]:
    manifest_path = profile / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = profile / filename
        if not path.is_file():
            raise ValueError(f"missing profile file: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"byte size mismatch for {path}")

    document_ids = _jsonl_ids(profile / "corpus.jsonl")
    query_ids = _jsonl_ids(profile / "queries.jsonl")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"duplicate document ids in {profile}")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"duplicate query ids in {profile}")

    document_id_set = set(document_ids)
    query_id_set = set(query_ids)
    qrel_count = 0
    judged_queries: set[str] = set()
    with (profile / "qrels.tsv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = row["query_id"]
            document_id = row["document_id"]
            relevance = int(row["relevance"])
            if query_id not in query_id_set:
                raise ValueError(f"unknown query id in qrels: {query_id}")
            if document_id not in document_id_set:
                raise ValueError(
                    f"unknown document id in qrels: {document_id}"
                )
            if relevance <= 0:
                raise ValueError("generated qrels must contain positive labels")
            judged_queries.add(query_id)
            qrel_count += 1
    if judged_queries != query_id_set:
        raise ValueError("every generated query must have a positive qrel")

    actual = {
        "documents": len(document_ids),
        "queries": len(query_ids),
        "qrels": qrel_count,
    }
    for key, value in actual.items():
        if manifest[key] != value:
            raise ValueError(
                f"manifest {key}={manifest[key]} does not match {value}"
            )
    return actual


def _jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(str(row["id"]))
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
