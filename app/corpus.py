from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.models import Document


def load_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            document = Document.model_validate(payload)
            if document.id in seen_ids:
                raise ValueError(
                    f"Duplicate document id {document.id!r} at line {line_number}"
                )
            seen_ids.add(document.id)
            documents.append(document)

    return documents


def write_jsonl(path: Path, documents: Iterable[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as corpus_file:
        for document in documents:
            corpus_file.write(document.model_dump_json() + "\n")
