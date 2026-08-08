import csv
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from scripts.validate_retrieval_corpora import validate_profile


def write_profile(root: Path, *, qrel_document_id: str = "d1") -> None:
    corpus = root / "corpus.jsonl"
    queries = root / "queries.jsonl"
    qrels = root / "qrels.tsv"
    root.mkdir()
    corpus.write_text(
        json.dumps({"id": "d1", "title": "one", "content": "text"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"id": "q1", "text": "query"}) + "\n",
        encoding="utf-8",
    )
    with qrels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "document_id", "relevance"])
        writer.writerow(["q1", qrel_document_id, 1])
    files = {}
    for path in (corpus, queries, qrels):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "documents": 1,
                "queries": 1,
                "qrels": 1,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_validate_profile_checks_hashes_counts_and_references(
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="corpus-validation-",
        dir=".pytest_cache",
    ) as temporary_directory:
        profile = Path(temporary_directory) / "profile"
        write_profile(profile)

        assert validate_profile(profile) == {
            "documents": 1,
            "queries": 1,
            "qrels": 1,
        }


def test_validate_profile_rejects_unknown_qrel_document(
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="corpus-validation-",
        dir=".pytest_cache",
    ) as temporary_directory:
        profile = Path(temporary_directory) / "profile"
        write_profile(profile, qrel_document_id="missing")

        with pytest.raises(ValueError, match="unknown document"):
            validate_profile(profile)
